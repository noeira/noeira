"""Device kernels for the FB training step.

Everything the CPU trainer does with host loops — gathering rows, packing
`[s | a | z]`, forming the twin-min target, the loss residuals, the gradient
sums — has to happen on device for M2, because the alternative is a per-step
host round trip. At `BATCH = 1024`, `D = 128` and walker's `OBS = 24`, packing
`[s|a|z]` on the host and uploading it costs ~650 KB per step; over 2 M steps
that is 1.3 TB across PCIe to avoid arithmetic that takes microseconds on the
GPU. So the dataset stays resident on device, the sampler writes indices on
device (`UniformDeviceSampler.draw_into_device`), and the batch is assembled
by the gather/pack kernels below without ever touching the host.

⚠ **Naive kernels on purpose.** One thread per output element, inner loops over
the contracted axis, no shared-memory tiling except in the reductions. Heavy
blocked kernels hard-crash the Metal compiler on Apple, and this file has to
compile on the laptop the parity gates run on even though the real target is
NVIDIA. The GEMM-shaped work lives in `PairwiseDot`, which carries the same
constraint for the same reason.

⚠ The reductions use the single-block grid-stride form (`block.sum` over a
`TPB_REDUCE` block, each thread striding the whole array) rather than a
multi-block partial-sum pass. Two reasons: it is one launch with no second
buffer, and it avoids a conditional read-modify-write across blocks — the
pattern this project has already been bitten by on NVIDIA, where CUDA drops
conditional RMW stores in reduction kernels.
"""

from std.math import sqrt, abs
from std.random import random_float64
from max.gpu import block_dim, block_idx, thread_idx, global_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT, TPB, TPB_REDUCE
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.ptr import mptr
from noeira.nn.random.box_muller import (
    box_muller_normal,
    box_muller_normal_gpu,
    box_muller_normal_gpu_dev,
    advance_rng_offset_kernel,
)
from std.random.philox import Random as PhiloxRandom
from noeira.data.resident import IDX_DT


# ══════════════════════════════════════════════════════════════════════
# Gather / pack
# ══════════════════════════════════════════════════════════════════════


def gather_rows_kernel[ROW_DIM: Int, BATCH: Int](
    src: Pointer[Scalar[DT], MutAnyOrigin],
    idx: Pointer[Scalar[IDX_DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`dst[i, d] = src[idx[i], d]`.

    Element-parallel, not lane-parallel: a per-lane kernel serialises the row
    copy inside each thread and launches only BATCH threads, which measured at
    ~73% of GPU time on wide rows elsewhere in this repo.
    """
    var t = Int(global_idx.x)
    if t >= BATCH * ROW_DIM:
        return
    var i = t // ROW_DIM
    var d = t % ROW_DIM
    dst[unsafe_offset=t] = src[unsafe_offset=Int(idx[unsafe_offset=i]) * ROW_DIM + d]


def gather_rows_into_kernel[SRC_DIM: Int, DST_STRIDE: Int, BATCH: Int](
    src: Pointer[Scalar[DT], MutAnyOrigin],
    idx: Pointer[Scalar[IDX_DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`dst[i, 0:SRC_DIM] = src[idx[i], :]` into rows of width `DST_STRIDE`.

    `gather_rows_kernel` assumes the destination row is exactly as wide as the
    source. That stops being true once part of a batch row is DERIVED rather
    than stored (§12.36): the ring holds `STORE_OBS` columns and the batch row
    is `OBS` wide, with `derive_tail_kernel` filling the rest.

    At `DST_STRIDE == SRC_DIM` this is `gather_rows_kernel` exactly, which is
    the case every caller that does not derive a tail still takes.
    """
    var t = Int(global_idx.x)
    if t >= BATCH * SRC_DIM:
        return
    var i = t // SRC_DIM
    var d = t % SRC_DIM
    dst[unsafe_offset=i * DST_STRIDE + d] = src[
        unsafe_offset=Int(idx[unsafe_offset=i]) * SRC_DIM + d
    ]


def derive_tail_kernel[
    ROWS: Int, CAP: Int, LANES: Int, STORE_OBS: Int, ACT: Int, TAIL: Int,
    OBS: Int,
](
    r_obs: Pointer[Scalar[DT], MutAnyOrigin],    # CAP x STORE_OBS
    r_act: Pointer[Scalar[DT], MutAnyOrigin],    # CAP x ACT
    r_age: Pointer[Scalar[DT], MutAnyOrigin],    # CAP, steps since this lane's reset
    idx: Pointer[Scalar[IDX_DT], MutAnyOrigin],  # ROWS, the drawn rows
    spec: Pointer[Scalar[DType.int32], MutAnyOrigin],   # TAIL x 4
    act_scale: Scalar[DT],
    act_clip: Scalar[DT],
    dst: Pointer[Scalar[DT], MutAnyOrigin],      # the batch, OBS-wide rows
    row0: Int32,
):
    """Fill the DERIVED TAIL of a gathered batch row from the ring itself.

    A replay row's `[STORE_OBS, OBS)` columns are not stored — they are a
    function of the SAME lane's earlier rows, so storing them is the
    redundancy §12.23 removed for `next_obs`, one layer up (docs §12.36). For
    the G1 that tail is `last_action 29 | history 372`: 401 floats per row
    against the ONE `r_age` float this needs, 4860 B/row against 3260.

    ⚠ Generic ON PURPOSE. The agent must not know what the tail MEANS — the
    layout is one env's business. `spec` carries it, four int32 per output
    element:

        [0] kind      0 = read `r_obs`, 1 = read `r_act` (scaled and clipped)
        [1] steps     lane-aligned back-steps: row - steps * LANES
        [2] src_off   column within that source row
        [3] min_age   written as 0 when `r_age[row] < min_age`

    The env builds it once on the host (`g1_build_tail_spec`) and hands it
    over. Zero cost when `TAIL == 0`: the caller does not launch this.

    ⚠ Looking BACKWARD needs no sampling-bound change — a predecessor row is
    always already written, unlike the successor `next_obs` needed.
    """
    var t = Int(global_idx.x)
    if t >= ROWS * TAIL:
        return
    var i = t // TAIL
    var k = t % TAIL
    var row = Int(idx[unsafe_offset=i])
    var age = Int(r_age[unsafe_offset=row])

    var o = (Int(row0) + i) * OBS + STORE_OBS + k
    if age < Int(spec[unsafe_offset=k * 4 + 3]):
        dst[unsafe_offset=o] = Scalar[DT](0.0)
        return

    var r = row - Int(spec[unsafe_offset=k * 4 + 1]) * LANES
    while r < 0:
        r += CAP
    var src_off = Int(spec[unsafe_offset=k * 4 + 2])
    if Int(spec[unsafe_offset=k * 4]) == 0:
        dst[unsafe_offset=o] = r_obs[unsafe_offset=r * STORE_OBS + src_off]
    else:
        var a = r_act[unsafe_offset=r * ACT + src_off] * act_scale
        if a > act_clip:
            a = act_clip
        if a < -act_clip:
            a = -act_clip
        dst[unsafe_offset=o] = a


def gather_idx_kernel[BATCH: Int](
    table: Pointer[Scalar[IDX_DT], MutAnyOrigin],
    idx: Pointer[Scalar[IDX_DT], MutAnyOrigin],
    dst: Pointer[Scalar[IDX_DT], MutAnyOrigin],
):
    """`dst[i] = table[idx[i]]` — remaps sampled rows through a lookup.

    Used for the `s'` indices: a precomputed `next_row` table encodes the
    episode boundaries, so `s'` is never the first row of the FOLLOWING
    episode. Doing it as a table lookup keeps the boundary logic on the host
    where it is written once, and off the hot path entirely.

    Separate from `gather_rows_kernel` because the index dtype is `IDX_DT`,
    not `DT`: routing indices through the float gather would lose exactness
    above 2^24 rows, which a 10 M-row dataset is not far from.
    """
    var i = Int(global_idx.x)
    if i < BATCH:
        dst[unsafe_offset=i] = table[unsafe_offset=Int(idx[unsafe_offset=i])]


def pack3_kernel[A_DIM: Int, B_DIM: Int, C_DIM: Int, BATCH: Int](
    a: Pointer[Scalar[DT], MutAnyOrigin],
    b: Pointer[Scalar[DT], MutAnyOrigin],
    c: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`dst[i] = [a[i] | b[i] | c[i]]` — the `[s | a | z]` row of the F net."""
    comptime W = A_DIM + B_DIM + C_DIM
    var t = Int(global_idx.x)
    if t >= BATCH * W:
        return
    var i = t // W
    var k = t % W
    if k < A_DIM:
        dst[unsafe_offset=t] = a[unsafe_offset=i * A_DIM + k]
    elif k < A_DIM + B_DIM:
        dst[unsafe_offset=t] = b[unsafe_offset=i * B_DIM + (k - A_DIM)]
    else:
        dst[unsafe_offset=t] = c[unsafe_offset=i * C_DIM + (k - A_DIM - B_DIM)]


def pack2_kernel[A_DIM: Int, B_DIM: Int, BATCH: Int](
    a: Pointer[Scalar[DT], MutAnyOrigin],
    b: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`dst[i] = [a[i] | b[i]]` — the `[s | z]` row of the actor."""
    comptime W = A_DIM + B_DIM
    var t = Int(global_idx.x)
    if t >= BATCH * W:
        return
    var i = t // W
    var k = t % W
    dst[unsafe_offset=t] = a[unsafe_offset=i * A_DIM + k] if k < A_DIM else b[unsafe_offset=i * B_DIM + (k - A_DIM)]


def slice_cols_kernel[
    SRC_W: Int, OFFSET: Int, OUT_W: Int, BATCH: Int
](
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`dst[i, k] = src[i, OFFSET + k]` — pulls the ACTION slice back out of
    the F net's input gradient for the actor update."""
    var t = Int(global_idx.x)
    if t >= BATCH * OUT_W:
        return
    var i = t // OUT_W
    var k = t % OUT_W
    dst[unsafe_offset=t] = src[unsafe_offset=i * SRC_W + OFFSET + k]


# ══════════════════════════════════════════════════════════════════════
# Elementwise
# ══════════════════════════════════════════════════════════════════════


def fill_kernel[N: Int](
    y: Pointer[Scalar[DT], MutAnyOrigin], v: Scalar[DT]
):
    var t = Int(global_idx.x)
    if t < N:
        y[unsafe_offset=t] = v


def axpy_kernel[N: Int](
    y: Pointer[Scalar[DT], MutAnyOrigin],
    x: Pointer[Scalar[DT], MutAnyOrigin],
    alpha: Scalar[DT],
):
    """`y += alpha * x`."""
    var t = Int(global_idx.x)
    if t < N:
        y[unsafe_offset=t] = y[unsafe_offset=t] + alpha * x[unsafe_offset=t]


def scale_kernel[N: Int](
    y: Pointer[Scalar[DT], MutAnyOrigin],
    x: Pointer[Scalar[DT], MutAnyOrigin],
    alpha: Scalar[DT],
):
    """`y = alpha * x`."""
    var t = Int(global_idx.x)
    if t < N:
        y[unsafe_offset=t] = alpha * x[unsafe_offset=t]


def sum3_scaled_kernel[N: Int](
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    a: Pointer[Scalar[DT], MutAnyOrigin],
    b: Pointer[Scalar[DT], MutAnyOrigin],
    c: Pointer[Scalar[DT], MutAnyOrigin],
    w: Scalar[DT],
):
    """`dst = a + b + w*c` — the three gradients arriving at `B(s+)`: one per
    twin from the measure loss, plus the orthonormality term."""
    var t = Int(global_idx.x)
    if t < N:
        dst[unsafe_offset=t] = a[unsafe_offset=t] + b[unsafe_offset=t] + w * c[unsafe_offset=t]


def pessimism_blend_kernel[N: Int](
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    m1: Pointer[Scalar[DT], MutAnyOrigin],
    m2: Pointer[Scalar[DT], MutAnyOrigin],
    gamma: Scalar[DT],
    penalty: Scalar[DT],
):
    """`dst = gamma * (mean(m1,m2) - penalty*|m1-m2|)`, entrywise.

    BFM-Zero's `get_targets_uncertainty` (`fb/agent.py:328`) for an ensemble
    of two. Its `preds_unc` sums `|p_i - p_j|` over all ordered pairs and
    divides by `P^2 - P`, which at P = 2 is exactly `|m1 - m2|`. So:

        penalty 0.0  ->  the MEAN
        penalty 0.5  ->  exactly min(m1, m2)

    ⚠ THE PENALTY IS PER-TARGET AND THEY ARE NOT THE SAME. The reference ships
    `fb_pessimism_penalty=0.0` for the FB measure target and 0.5 for all three
    Q critics (`train.py:638-641`) — pessimism is a VALUE-function device and
    a successor MEASURE is not a value, so biasing it down has no
    justification. We shipped `min` at both, i.e. 0.5 on the FB target, which
    biased every one of BATCH^2 entries down by half the ensemble disagreement
    at each of ~6 M gradient steps (§12.15).
    """
    var t = Int(global_idx.x)
    if t >= N:
        return
    var a = m1[unsafe_offset=t]
    var b = m2[unsafe_offset=t]
    var lo = a if a < b else b
    var hi = b if a < b else a
    # `mean - p*|a-b|` rearranged as `lo*(0.5+p) + hi*(0.5-p)`. Algebraically
    # the same; in float32 it is NOT. The literal form subtracts two rounded
    # halves and cancels — measured 9.5e-07 off the exact min at p = 0.5,
    # which is precisely the setting Q_D runs at. This form makes p = 0.5
    # return `lo` BIT-EXACTLY (the weights are 1 and 0) and is well
    # conditioned in between.
    dst[unsafe_offset=t] = gamma * (
        lo * (Scalar[DT](0.5) + penalty) + hi * (Scalar[DT](0.5) - penalty)
    )


def mean_abs_into_kernel[N: Int](
    x: Pointer[Scalar[DT], MutAnyOrigin],
    acc: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`acc[0] = mean(|x|)` — NOT `|mean(x)|`. See `mean_abs_into_t`."""
    var s = Scalar[DT](0)
    for i in range(N):
        var v = x[unsafe_offset=i]
        s += -v if v < Scalar[DT](0) else v
    acc[unsafe_offset=0] = s / Scalar[DT](N)


def residual_grad_kernel[N: Int](
    go: Pointer[Scalar[DT], MutAnyOrigin],
    m: Pointer[Scalar[DT], MutAnyOrigin],
    mt: Pointer[Scalar[DT], MutAnyOrigin],
    inv_n: Scalar[DT],
):
    """`go = 2·(m - mt)·inv_n` — the upstream gradient of `mean((m-mt)^2)`.

    `inv_n` is `1/BATCH^2`, passed in rather than recomputed so the CPU and GPU
    paths divide by exactly the same constant.
    """
    var t = Int(global_idx.x)
    if t < N:
        go[unsafe_offset=t] = Scalar[DT](2.0) * (m[unsafe_offset=t] - mt[unsafe_offset=t]) * inv_n


def pessimism_row_weights_kernel[N: Int](
    w1: Pointer[Scalar[DT], MutAnyOrigin],
    w2: Pointer[Scalar[DT], MutAnyOrigin],
    a: Pointer[Scalar[DT], MutAnyOrigin],
    b: Pointer[Scalar[DT], MutAnyOrigin],
    penalty: Scalar[DT],
):
    """The partials d/da and d/db of `mean(a,b) - penalty*|a-b|`, per row.

    The same reduction `pessimism_blend_kernel` applies to the VALUE, applied
    to its GRADIENT: written as `lo*(0.5+p) + hi*(0.5-p)`, the derivative is
    `0.5+p` on whichever of the two is smaller and `0.5-p` on the other. At
    `penalty = 0.5` that is 1 and 0 — the whole gradient goes to the min
    branch, which is what `actor_pessimism_penalty=0.5` means.
    """
    var t = Int(global_idx.x)
    if t >= N:
        return
    var lo_w = Scalar[DT](0.5) + penalty
    var hi_w = Scalar[DT](0.5) - penalty
    if a[unsafe_offset=t] <= b[unsafe_offset=t]:
        w1[unsafe_offset=t] = lo_w
        w2[unsafe_offset=t] = hi_w
    else:
        w1[unsafe_offset=t] = hi_w
        w2[unsafe_offset=t] = lo_w


def scale_rows_kernel[N: Int, DIM: Int](
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    src: Pointer[Scalar[DT], MutAnyOrigin],
    rowscale: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`dst[i, :] = rowscale[i] * src[i, :]`."""
    var t = Int(global_idx.x)
    if t >= N * DIM:
        return
    dst[unsafe_offset=t] = rowscale[unsafe_offset=t // DIM] * src[unsafe_offset=t]


def fb_diag_override_kernel[BATCH: Int](
    go: Pointer[Scalar[DT], MutAnyOrigin],
    diag_scale: Scalar[DT],
):
    """Overwrite `go[i,i]` with `diag_scale` on a `[BATCH, BATCH]` matrix.

    The reference splits `M`'s diagonal out of the squared term entirely
    (`(diff * off_diag).pow(2)`) and gives it the LINEAR anchor instead
    (`-diagonal(diff).mean()`). So the diagonal's upstream gradient is a
    CONSTANT `-2/BATCH` (our 2x scale), not `2*(M-Mt)/...`: the residual
    never reaches it. Run `residual_grad_kernel` first, then this.
    """
    var i = Int(global_idx.x)
    if i < BATCH:
        go[unsafe_offset=i * BATCH + i] = diag_scale


def diag_sumsq_kernel[BATCH: Int](
    m: Pointer[Scalar[DT], MutAnyOrigin],
    acc: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`acc[0] = sum_i m[i,i]^2` on a `[BATCH, BATCH]` matrix.

    `L_ortho`'s quadratic runs over the OFF-DIAGONAL only (`agent.py:249`
    masks `Cov` with `off_diag`), and the cheap way to that sum is the full
    sum of squares minus the diagonal's own. Unlike `fb_diag_stats_kernel`
    there is no target to subtract here: the ortho target IS zero, so the
    residual is `O` itself.
    """
    var t = Int(thread_idx.x)
    var s2: Scalar[DT] = 0.0
    var k = t
    while k < BATCH:
        var v = m[unsafe_offset=k * BATCH + k]
        s2 += v * v
        k += TPB_REDUCE
    var t2 = block.sum[block_size=TPB_REDUCE, broadcast=False](val=s2)
    if t == 0:
        acc[unsafe_offset=0] = t2[0]


def fb_diag_stats_kernel[BATCH: Int](
    m: Pointer[Scalar[DT], MutAnyOrigin],
    mt: Pointer[Scalar[DT], MutAnyOrigin],
    acc: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`acc[0] = sum_i r_ii`, `acc[1] = sum_i r_ii^2` for `r = m - mt`.

    Both halves of the loss VALUE need the diagonal separated: the anchor is
    the diagonal mean, and the off-diagonal sum of squares is the full sum
    minus the diagonal's own. Reducing once for both keeps them consistent.
    """
    var t = Int(thread_idx.x)
    var s1: Scalar[DT] = 0.0
    var s2: Scalar[DT] = 0.0
    var k = t
    while k < BATCH:
        var r = m[unsafe_offset=k * BATCH + k] - mt[unsafe_offset=k * BATCH + k]
        s1 += r
        s2 += r * r
        k += TPB_REDUCE
    var t1 = block.sum[block_size=TPB_REDUCE, broadcast=False](val=s1)
    var t2 = block.sum[block_size=TPB_REDUCE, broadcast=False](val=s2)
    if t == 0:
        acc[unsafe_offset=0] = t1[0]
        acc[unsafe_offset=1] = t2[0]


def sq_diff_reduce_kernel[N: Int](
    m: Pointer[Scalar[DT], MutAnyOrigin],
    mt: Pointer[Scalar[DT], MutAnyOrigin],
    acc: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`acc[0] = mean((m - mt)^2)` over `[N]`. ONE block, grid-stride."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < N:
        var r = m[unsafe_offset=k] - mt[unsafe_offset=k]
        my_sum += r * r
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[unsafe_offset=0] = total[0] / Scalar[DT](N)


def scale_by_inv_mag_kernel[N: Int](
    y: Pointer[Scalar[DT], MutAnyOrigin],
    x: Pointer[Scalar[DT], MutAnyOrigin],
    mag: Pointer[Scalar[DT], MutAnyOrigin],
    base: Scalar[DT],
    eps: Scalar[DT],
):
    """`y = (base / max(|mag[0]|, eps)) * x` — TD3+BC's adaptive scale, with
    the magnitude read FROM DEVICE.

    ⚠⚠ Exists because the host form was silently conditional on logging. The
    trainer computed `lam = 1/|loss|` only when `want_loss` was true — one step
    in `LOG_EVERY` — and ran `lam = 1.0` otherwise; under CUDA-graph capture
    (`want_loss=False` always) it was NEVER applied. Reading the magnitude from
    a device buffer makes the scale unconditional AND capture-safe, because
    nothing crosses to the host.
    """
    var t = Int(global_idx.x)
    if t >= N:
        return
    var m = mag[unsafe_offset=0]
    if m < 0:
        m = -m
    if m < eps:
        m = eps
    y[unsafe_offset=t] = (base / m) * x[unsafe_offset=t]


def sum_reduce_kernel[N: Int](
    x: Pointer[Scalar[DT], MutAnyOrigin],
    acc: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`acc[0] = mean(x)` over `[N]`. ONE block, grid-stride."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < N:
        my_sum += x[unsafe_offset=k]
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[unsafe_offset=0] = total[0] / Scalar[DT](N)


def sumsq_reduce_kernel[N: Int](
    x: Pointer[Scalar[DT], MutAnyOrigin],
    acc: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`acc[0] = mean(x^2)` over `[N]`. ONE block, grid-stride."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < N:
        var v = x[unsafe_offset=k]
        my_sum += v * v
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[unsafe_offset=0] = total[0] / Scalar[DT](N)


def smooth_action_kernel[N: Int](
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    pi: Pointer[Scalar[DT], MutAnyOrigin],
    noise: Pointer[Scalar[DT], MutAnyOrigin],
    sigma: Scalar[DT],
    clip: Scalar[DT],
):
    """TD3 target-policy smoothing: `clamp(pi + clamp(sigma·n, ±clip), ±1)`."""
    var t = Int(global_idx.x)
    if t >= N:
        return
    var n = noise[unsafe_offset=t] * sigma
    if n > clip:
        n = clip
    elif n < -clip:
        n = -clip
    var v = pi[unsafe_offset=t] + n
    if v > Scalar[DT](1.0):
        v = Scalar[DT](1.0)
    elif v < Scalar[DT](-1.0):
        v = Scalar[DT](-1.0)
    dst[unsafe_offset=t] = v


def hinge_axpy_kernel[N: Int](
    y: Pointer[Scalar[DT], MutAnyOrigin],
    x: Pointer[Scalar[DT], MutAnyOrigin],
    alpha: Scalar[DT],
    margin: Scalar[DT],
):
    """`y += alpha · sign(x) · max(|x| - margin, 0)` — the gradient of the
    hinged action penalty `mean(relu(|pi| - margin)^2)`, zero inside the
    band. See `FBTrainer.act_l2_margin` for why the band matters."""
    var t = Int(global_idx.x)
    if t >= N:
        return
    var v = x[unsafe_offset=t]
    var a = v if v >= 0 else -v
    var e = a - margin
    if e <= Scalar[DT](0):
        return
    var g = e if v >= 0 else -e
    y[unsafe_offset=t] = y[unsafe_offset=t] + alpha * g


def masked_rows_axpy_kernel[BATCH: Int, W: Int](
    y: Pointer[Scalar[DT], MutAnyOrigin],
    x: Pointer[Scalar[DT], MutAnyOrigin],
    mask: Pointer[Scalar[DT], MutAnyOrigin],
    alpha: Scalar[DT],
):
    """`y[i, :] += alpha · mask[i] · x[i, :]` — a per-ROW weighted axpy. The
    BC term's form once a batch mixes rows that have a data action to clone
    (mask 1) with rows that do not (mask 0)."""
    var t = Int(global_idx.x)
    if t >= BATCH * W:
        return
    var i = t // W
    y[unsafe_offset=t] = y[unsafe_offset=t] + alpha * mask[unsafe_offset=i] * x[unsafe_offset=t]


def project_sphere_kernel[D: Int, BATCH: Int](
    z: Pointer[Scalar[DT], MutAnyOrigin], radius: Scalar[DT]
):
    """Rescale each row of `z` onto the radius-`sqrt(D)` sphere, one thread per
    ROW.

    ⚠ This is the device twin of `z_sampler._project_to_sphere`, and it exists
    for the same reason that one does: a `z` off the sphere crashes nothing and
    trains to a policy that emits plausible garbage. Both must agree, so the
    degenerate-row rule is the same — a row with no direction becomes `radius`
    on the first axis rather than being rescaled by `radius/~0`, which would
    amplify rounding noise into a full-magnitude z pointing nowhere.
    """
    var i = Int(global_idx.x)
    if i >= BATCH:
        return
    var base = i * D
    var acc: Scalar[DT] = 0.0
    for k in range(D):
        var v = z[unsafe_offset=base + k]
        acc += v * v
    var n = sqrt(acc)
    if n < Scalar[DT](1e-12):
        for k in range(D):
            z[unsafe_offset=base + k] = Scalar[DT](0)
        z[unsafe_offset=base] = radius
        return
    var s = radius / n
    for k in range(D):
        z[unsafe_offset=base + k] = z[unsafe_offset=base + k] * s


def z_mixture_kernel[D: Int, BATCH: Int](
    z: Pointer[Scalar[DT], MutAnyOrigin],
    gauss: Pointer[Scalar[DT], MutAnyOrigin],
    b_states: Pointer[Scalar[DT], MutAnyOrigin],
    pick: Pointer[Scalar[DT], MutAnyOrigin],
    uniform_frac: Scalar[DT],
    n_b_rows: Int32,
):
    """Half the rows Gaussian, half copied from `B(s+)`. One thread per ROW.

    `pick[i]` carries two independent uniforms per row: `pick[2i]` chooses the
    branch, `pick[2i+1]` chooses which `B` row to copy. Drawing them outside
    keeps this kernel free of RNG state, so the SAME buffer replays identically
    — which is what makes a CPU/GPU parity gate on the mixture possible at all.

    Renormalisation is NOT done here — `project_sphere_kernel` runs immediately
    after, unconditionally, on both branches.

    ⚠ `n_b_rows` is `Int32`, not `Int`, because a kernel's runtime scalar args
    must be FIXED-WIDTH: nightly rejects `Int`/`UInt` with "do not conform to
    DevicePassable". It was `Int` and compiled until the toolchain tightened,
    so the failure appears at the CALL SITE (`enqueue_function`) as a chain of
    "function instantiation failed" notes ending in that constraint — nothing
    points at this line. Widen to `Int` once inside the body; keep the ABI
    fixed-width.
    """
    var i = Int(global_idx.x)
    if i >= BATCH:
        return
    var base = i * D
    var nb = Int(n_b_rows)
    var use_uniform = nb <= 0 or pick[unsafe_offset=2 * i] < uniform_frac
    if use_uniform:
        for k in range(D):
            z[unsafe_offset=base + k] = gauss[unsafe_offset=base + k]
    else:
        var src = Int(pick[unsafe_offset=2 * i + 1] * Scalar[DT](nb))
        if src >= nb:
            src = nb - 1
        if src < 0:
            src = 0
        for k in range(D):
            z[unsafe_offset=base + k] = b_states[unsafe_offset=src * D + k]


def uniform01_kernel[N: Int](
    dst: Pointer[Scalar[DT], MutAnyOrigin], seed: UInt64, offset: UInt64
):
    """`dst[i] ~ U[0, 1)`, Philox, host offset — the draw `z_mixture_kernel`'s
    `pick` buffer needs.

    ⚠⚠ Exists because both GPU training scripts filled `pick` with
    `box_muller_normal_gpu` — GAUSSIANS — for the whole of M2 and the A2
    sweep (found 2026-09-07 while writing the online agent). Against
    `uniform_frac = 0.5` a N(0,1) draw takes the uniform branch 69 % of the
    time, and `Int(n · nb)` on a negative `n` clamps to row 0, so about half
    of the `B(s+)` picks were the SAME row. Nothing raised, `|B|` stayed
    pinned, the loss descended. Every §13 number and every A2 arm trained
    under that mixture; A2's deltas are between arms that share it.
    """
    var i = Int(global_idx.x)
    if i >= N:
        return
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset)
    dst[unsafe_offset=i] = Scalar[DT](Float32(philox.step_uniform()[0]))


# ══════════════════════════════════════════════════════════════════════
# Target-parameterized host-side ops.
#
# These exist so `FBTrainer.train_step` is written ONCE rather than as a CPU
# body and a GPU body that drift apart. Each pairs a host loop with the launch
# of its kernel above; the trainer never writes `comptime if target ==` itself.
#
# Grid dims are computed here, in one place, so a kernel whose element count is
# BATCH*W is never launched with BATCH blocks by mistake.
# ══════════════════════════════════════════════════════════════════════


def _blocks(n: Int) -> Int:
    return (n + TPB - 1) // TPB


def ensure_t[target: StaticString](
    mut t: Tensor, n: Int, ctx: Optional[DeviceContext] = None
) raises:
    """Size a tensor for `target`. On GPU the HOST mirror is sized too — the
    diagnostics and the parity gate read `.data`, and `ensure_gpu` alone leaves
    it empty (a fill would then index out of bounds)."""
    comptime if target == "cpu":
        t.ensure(n)
    else:
        t.ensure(n)
        t.ensure_gpu(ctx.value(), n)


def pack3_t[
    target: StaticString, A_DIM: Int, B_DIM: Int, C_DIM: Int, BATCH: Int
](
    mut dst: Tensor, mut a: Tensor, mut b: Tensor, mut c: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """`dst[i] = [a[i] | b[i] | c[i]]`."""
    comptime W = A_DIM + B_DIM + C_DIM
    ensure_t[target](dst, BATCH * W, ctx)
    comptime if target == "cpu":
        for i in range(BATCH):
            var o = i * W
            for k in range(A_DIM):
                dst.data[o + k] = a.data[i * A_DIM + k]
            for k in range(B_DIM):
                dst.data[o + A_DIM + k] = b.data[i * B_DIM + k]
            for k in range(C_DIM):
                dst.data[o + A_DIM + B_DIM + k] = c.data[i * C_DIM + k]
    else:
        var d = ctx.value()
        d.enqueue_function[pack3_kernel[A_DIM, B_DIM, C_DIM, BATCH]](
            a.dev.value().unsafe_ptr(), b.dev.value().unsafe_ptr(),
            c.dev.value().unsafe_ptr(), dst.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * W), block_dim=TPB,
        )


def pack2_t[
    target: StaticString, A_DIM: Int, B_DIM: Int, BATCH: Int
](
    mut dst: Tensor, mut a: Tensor, mut b: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """`dst[i] = [a[i] | b[i]]`."""
    comptime W = A_DIM + B_DIM
    ensure_t[target](dst, BATCH * W, ctx)
    comptime if target == "cpu":
        for i in range(BATCH):
            var o = i * W
            for k in range(A_DIM):
                dst.data[o + k] = a.data[i * A_DIM + k]
            for k in range(B_DIM):
                dst.data[o + A_DIM + k] = b.data[i * B_DIM + k]
    else:
        var d = ctx.value()
        d.enqueue_function[pack2_kernel[A_DIM, B_DIM, BATCH]](
            a.dev.value().unsafe_ptr(), b.dev.value().unsafe_ptr(),
            dst.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * W), block_dim=TPB,
        )


def axpy_t[target: StaticString, N: Int](
    mut y: Tensor, mut x: Tensor, alpha: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`y += alpha * x`."""
    comptime if target == "cpu":
        for i in range(N):
            y.data[i] = y.data[i] + alpha * x.data[i]
    else:
        var d = ctx.value()
        d.enqueue_function[axpy_kernel[N]](
            y.dev.value().unsafe_ptr(), x.dev.value().unsafe_ptr(), alpha,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def hinge_axpy_t[target: StaticString, N: Int](
    mut y: Tensor, mut x: Tensor, alpha: Scalar[DT], margin: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`y += alpha · sign(x) · relu(|x| - margin)`."""
    comptime if target == "cpu":
        for i in range(N):
            var v = x.data[i]
            var a = v if v >= 0 else -v
            var e = a - margin
            if e > Scalar[DT](0):
                y.data[i] = y.data[i] + alpha * (e if v >= 0 else -e)
    else:
        var d = ctx.value()
        d.enqueue_function[hinge_axpy_kernel[N]](
            y.dev.value().unsafe_ptr(), x.dev.value().unsafe_ptr(), alpha, margin,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def masked_rows_axpy_t[target: StaticString, BATCH: Int, W: Int](
    mut y: Tensor, mut x: Tensor, mut mask: Tensor, alpha: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`y[i, :] += alpha · mask[i] · x[i, :]`."""
    comptime if target == "cpu":
        for i in range(BATCH):
            var m = alpha * mask.data[i]
            for k in range(W):
                y.data[i * W + k] = y.data[i * W + k] + m * x.data[i * W + k]
    else:
        var d = ctx.value()
        d.enqueue_function[masked_rows_axpy_kernel[BATCH, W]](
            y.dev.value().unsafe_ptr(), x.dev.value().unsafe_ptr(),
            mask.dev.value().unsafe_ptr(), alpha,
            grid_dim=_blocks(BATCH * W), block_dim=TPB,
        )


def scale_t[target: StaticString, N: Int](
    mut y: Tensor, mut x: Tensor, alpha: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`y = alpha * x`."""
    ensure_t[target](y, N, ctx)
    comptime if target == "cpu":
        for i in range(N):
            y.data[i] = alpha * x.data[i]
    else:
        var d = ctx.value()
        d.enqueue_function[scale_kernel[N]](
            y.dev.value().unsafe_ptr(), x.dev.value().unsafe_ptr(), alpha,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def sum3_scaled_t[target: StaticString, N: Int](
    mut dst: Tensor, mut a: Tensor, mut b: Tensor, mut c: Tensor,
    w: Scalar[DT], ctx: Optional[DeviceContext] = None,
) raises:
    """`dst = a + b + w*c`."""
    ensure_t[target](dst, N, ctx)
    comptime if target == "cpu":
        for i in range(N):
            dst.data[i] = a.data[i] + b.data[i] + w * c.data[i]
    else:
        var d = ctx.value()
        d.enqueue_function[sum3_scaled_kernel[N]](
            dst.dev.value().unsafe_ptr(), a.dev.value().unsafe_ptr(),
            b.dev.value().unsafe_ptr(), c.dev.value().unsafe_ptr(), w,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def pessimism_row_weights_t[target: StaticString, N: Int](
    mut w1: Tensor, mut w2: Tensor, mut a: Tensor, mut b: Tensor,
    penalty: Scalar[DT], ctx: Optional[DeviceContext] = None,
) raises:
    """Per-row gradient weights of the pessimistic blend. See the kernel."""
    ensure_t[target](w1, N, ctx)
    ensure_t[target](w2, N, ctx)
    var lo_w = Scalar[DT](0.5) + penalty
    var hi_w = Scalar[DT](0.5) - penalty
    comptime if target == "cpu":
        for i in range(N):
            if a.data[i] <= b.data[i]:
                w1.data[i] = lo_w
                w2.data[i] = hi_w
            else:
                w1.data[i] = hi_w
                w2.data[i] = lo_w
    else:
        var d = ctx.value()
        d.enqueue_function[pessimism_row_weights_kernel[N]](
            w1.dev.value().unsafe_ptr(), w2.dev.value().unsafe_ptr(),
            a.dev.value().unsafe_ptr(), b.dev.value().unsafe_ptr(), penalty,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def scale_rows_t[target: StaticString, N: Int, DIM: Int](
    mut dst: Tensor, mut src: Tensor, mut rowscale: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """`dst[i, :] = rowscale[i] * src[i, :]`."""
    ensure_t[target](dst, N * DIM, ctx)
    comptime if target == "cpu":
        for i in range(N):
            var r = rowscale.data[i]
            for k in range(DIM):
                dst.data[i * DIM + k] = r * src.data[i * DIM + k]
    else:
        var d = ctx.value()
        d.enqueue_function[scale_rows_kernel[N, DIM]](
            dst.dev.value().unsafe_ptr(), src.dev.value().unsafe_ptr(),
            rowscale.dev.value().unsafe_ptr(),
            grid_dim=_blocks(N * DIM), block_dim=TPB,
        )


def pessimism_blend_t[target: StaticString, N: Int](
    mut dst: Tensor, mut m1: Tensor, mut m2: Tensor, gamma: Scalar[DT],
    penalty: Scalar[DT], ctx: Optional[DeviceContext] = None,
) raises:
    """`dst = gamma * (mean(m1,m2) - penalty*|m1-m2|)`. See the kernel."""
    ensure_t[target](dst, N, ctx)
    comptime if target == "cpu":
        for i in range(N):
            var a = m1.data[i]
            var b = m2.data[i]
            var lo = a if a < b else b
            var hi = b if a < b else a
            dst.data[i] = gamma * (
                lo * (Scalar[DT](0.5) + penalty)
                + hi * (Scalar[DT](0.5) - penalty)
            )
    else:
        var d = ctx.value()
        d.enqueue_function[pessimism_blend_kernel[N]](
            dst.dev.value().unsafe_ptr(), m1.dev.value().unsafe_ptr(),
            m2.dev.value().unsafe_ptr(), gamma, penalty,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def mean_abs_into_t[target: StaticString, N: Int](
    mut x: Tensor, mut acc: Tensor, ctx: Optional[DeviceContext] = None
) raises:
    """`acc[0] = mean(|x|)` — NO download, so it is capture-safe.

    ⚠ NOT `|mean(x)|`. `scale_reg` and TD3+BC's adaptive scale both weight by
    `Q.abs().mean()`; we weighted by `|Q.mean()|`, which is Jensen-smaller and
    collapses toward 0 as `Q` becomes sign-balanced — so the CPR style term
    faded out exactly as `|F|` grew (§12.15).
    """
    ensure_t[target](acc, 1, ctx)
    comptime if target == "cpu":
        var s = Float64(0)
        for i in range(N):
            var v = Float64(x.data[i])
            s += -v if v < 0.0 else v
        acc.data[0] = Scalar[DT](s / Float64(N))
    else:
        var d = ctx.value()
        d.enqueue_function[mean_abs_into_kernel[N]](
            x.dev.value().unsafe_ptr(), acc.dev.value().unsafe_ptr(),
            grid_dim=1, block_dim=1,
        )


def smooth_action_t[target: StaticString, N: Int](
    mut dst: Tensor, mut pi: Tensor, mut noise: Tensor,
    sigma: Scalar[DT], clip: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`clamp(pi + clamp(sigma·noise, ±clip), ±1)`."""
    ensure_t[target](dst, N, ctx)
    comptime if target == "cpu":
        for i in range(N):
            var n = noise.data[i] * sigma
            if n > clip:
                n = clip
            elif n < -clip:
                n = -clip
            var v = pi.data[i] + n
            if v > Scalar[DT](1.0):
                v = Scalar[DT](1.0)
            elif v < Scalar[DT](-1.0):
                v = Scalar[DT](-1.0)
            dst.data[i] = v
    else:
        var d = ctx.value()
        d.enqueue_function[smooth_action_kernel[N]](
            dst.dev.value().unsafe_ptr(), pi.dev.value().unsafe_ptr(),
            noise.dev.value().unsafe_ptr(), sigma, clip,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def slice_cols_t[
    target: StaticString, SRC_W: Int, OFFSET: Int, OUT_W: Int, BATCH: Int
](
    mut dst: Tensor, mut src: Tensor, ctx: Optional[DeviceContext] = None
) raises:
    """`dst[i, k] = src[i, OFFSET + k]`."""
    ensure_t[target](dst, BATCH * OUT_W, ctx)
    comptime if target == "cpu":
        for i in range(BATCH):
            for k in range(OUT_W):
                dst.data[i * OUT_W + k] = src.data[i * SRC_W + OFFSET + k]
    else:
        var d = ctx.value()
        d.enqueue_function[slice_cols_kernel[SRC_W, OFFSET, OUT_W, BATCH]](
            src.dev.value().unsafe_ptr(), dst.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * OUT_W), block_dim=TPB,
        )


def mean_sq_t[target: StaticString, N: Int](
    mut x: Tensor, mut acc: Tensor, ctx: Optional[DeviceContext] = None
) raises -> Float64:
    """`mean(x^2)` over `[N]`. GPU path costs a device sync — diagnostics only.
    """
    comptime if target == "cpu":
        var s = Float64(0)
        for i in range(N):
            var v = Float64(x.data[i])
            s += v * v
        return s / Float64(N)
    else:
        var d = ctx.value()
        d.enqueue_function[sumsq_reduce_kernel[N]](
            x.dev.value().unsafe_ptr(), acc.dev.value().unsafe_ptr(),
            grid_dim=1, block_dim=TPB_REDUCE,
        )
        acc.download(d)
        return Float64(acc.data[0])


def mean_sq_into_t[target: StaticString, N: Int](
    mut x: Tensor, mut acc: Tensor, ctx: Optional[DeviceContext] = None
) raises:
    """`acc[0] = mean(x^2)` — NO download, capture-safe. `mean_sq_t` is the
    syncing sibling; use this one inside the train step and read `acc` at
    flush cadence."""
    ensure_t[target](acc, 1, ctx)
    comptime if target == "cpu":
        var s = Float64(0)
        for i in range(N):
            var v = Float64(x.data[i])
            s += v * v
        acc.data[0] = Scalar[DT](s / Float64(N))
    else:
        var d = ctx.value()
        d.enqueue_function[sumsq_reduce_kernel[N]](
            x.dev.value().unsafe_ptr(), acc.dev.value().unsafe_ptr(),
            grid_dim=1, block_dim=TPB_REDUCE,
        )


def mean_into_t[target: StaticString, N: Int](
    mut x: Tensor, mut acc: Tensor, ctx: Optional[DeviceContext] = None
) raises:
    """`acc[0] = mean(x)` — NO download, so it is capture-safe. The sibling
    `mean_t` returns the value to the host and therefore syncs; use this one
    anywhere inside the train step."""
    ensure_t[target](acc, 1, ctx)
    comptime if target == "cpu":
        var s = Float64(0)
        for i in range(N):
            s += Float64(x.data[i])
        acc.data[0] = Scalar[DT](s / Float64(N))
    else:
        var d = ctx.value()
        d.enqueue_function[sum_reduce_kernel[N]](
            x.dev.value().unsafe_ptr(), acc.dev.value().unsafe_ptr(),
            grid_dim=1, block_dim=TPB_REDUCE,
        )


def scale_by_inv_mag_t[target: StaticString, N: Int](
    mut y: Tensor, mut x: Tensor, mut mag: Tensor,
    base: Scalar[DT], eps: Scalar[DT] = 1e-6,
    ctx: Optional[DeviceContext] = None,
) raises:
    """`y = (base / max(|mag[0]|, eps)) * x`, with `mag` staying on device."""
    ensure_t[target](y, N, ctx)
    comptime if target == "cpu":
        var m = abs(Float64(mag.data[0]))
        if m < Float64(eps):
            m = Float64(eps)
        var a = Scalar[DT](Float64(base) / m)
        for i in range(N):
            y.data[i] = a * x.data[i]
    else:
        var d = ctx.value()
        d.enqueue_function[scale_by_inv_mag_kernel[N]](
            y.dev.value().unsafe_ptr(), x.dev.value().unsafe_ptr(),
            mag.dev.value().unsafe_ptr(), base, eps,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def mean_t[target: StaticString, N: Int](
    mut x: Tensor, mut acc: Tensor, ctx: Optional[DeviceContext] = None
) raises -> Float64:
    """`mean(x)` over `[N]`. GPU path costs a device sync — diagnostics only."""
    comptime if target == "cpu":
        var s = Float64(0)
        for i in range(N):
            s += Float64(x.data[i])
        return s / Float64(N)
    else:
        var d = ctx.value()
        d.enqueue_function[sum_reduce_kernel[N]](
            x.dev.value().unsafe_ptr(), acc.dev.value().unsafe_ptr(),
            grid_dim=1, block_dim=TPB_REDUCE,
        )
        acc.download(d)
        return Float64(acc.data[0])


def gaussian_t[target: StaticString, N: Int](
    mut t: Tensor, seed: UInt64, offset: UInt64,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Fill `[N]` with iid N(0,1).

    ⚠ CPU and GPU draw from DIFFERENT generators (host RNG vs Philox), so a
    CPU/GPU parity gate on the trainer must either zero the exploration noise
    or compare distributions rather than values. The alternative — a host draw
    uploaded every step — would put a PCIe round trip in the hot loop to make a
    test easier, which is the wrong trade.
    """
    ensure_t[target](t, N, ctx)
    comptime if target == "cpu":
        box_muller_normal(t.data.unsafe_ptr(), N)
    else:
        var d = ctx.value()
        box_muller_normal_gpu[N](
            d, mptr(t.dev.value().unsafe_ptr()), seed, offset
        )


def gaussian_dev_t[target: StaticString, N: Int](
    mut t: Tensor,
    seed: UInt64,
    ref [MutAnyOrigin] offset_buf: DeviceBuffer[DType.uint64],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`gaussian_t` with the Philox offset held ON DEVICE — the CUDA-graph-safe
    draw.

    ⚠⚠ This exists because the host-offset form is a CAPTURE TRAP. `gaussian_t`
    takes `offset: UInt64` by value, so capturing a step that calls it BAKES the
    offset into the graph: every replay draws the IDENTICAL noise, forever, and
    nothing raises. The training curve still descends — TD3 target smoothing
    with frozen noise is just a slightly different (deterministic) regulariser —
    so this fails silently in exactly the way `USE_ENV_CUDA_GRAPH` froze the
    physics step and every GPU example trained against a stopped simulator.

    The offset lives in a 1-element device buffer, is READ by the draw kernel
    and BUMPED by `advance_rng_offset_kernel` immediately after, so the whole
    advance is inside the captured sequence and each replay moves the stream.

    CPU target ignores `offset_buf` and draws from the host RNG, matching
    `gaussian_t`.
    """
    ensure_t[target](t, N, ctx)
    comptime if target == "cpu":
        box_muller_normal(t.data.unsafe_ptr(), N)
    else:
        var d = ctx.value()
        var off = LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin](
            mptr(offset_buf.unsafe_ptr())
        )
        box_muller_normal_gpu_dev[N](
            d, mptr(t.dev.value().unsafe_ptr()), seed, off
        )
        # ⚠ AFTER the draw, and with the same rounding the host path used
        # (`N + N % 2`), so a captured run and an eager run walk the SAME
        # Philox stream rather than diverging by a half-pair every step.
        comptime AMT = N + (N % 2)
        d.enqueue_function[advance_rng_offset_kernel[AMT]](
            off, grid_dim=1, block_dim=1
        )


# ══════════════════════════════════════════════════════════════════════
# A4 / FB-CPR additions (`fb/cpr.mojo`). Same conventions as above: naive
# one-thread-per-element kernels, and a `_t` host twin for each so the CPR
# trainer is written once for both targets.
# ══════════════════════════════════════════════════════════════════════


def uniform01_dev_kernel[N: Int](
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    seed: UInt64,
    offset_buf: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
):
    """`dst[i] ~ U[0, 1)`, Philox, offset read FROM DEVICE (capture-safe).
    Device-offset twin of `uniform01_kernel` — see its docstring for why the
    mixture kernels must be fed UNIFORMS. Lived in `online.mojo` first; the
    CPR trainer's interpolation weights need it too, so it is here."""
    var i = Int(global_idx.x)
    if i >= N:
        return
    var philox = PhiloxRandom(
        seed=seed + UInt64(i), offset=rebind[UInt64](offset_buf[0])
    )
    dst[unsafe_offset=i] = Scalar[DT](Float32(philox.step_uniform()[0]))


def window_mean_kernel[SEQ: Int, D: Int, NW: Int](
    src: Pointer[Scalar[DT], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`dst[w·SEQ + j, k] = mean_{j'} src[w·SEQ + j', k]` for every `j` —
    the per-window mean of `SEQ` consecutive rows, written back to ALL
    `SEQ` rows of the window (`repeat_interleave`). One thread per
    `(window, k)`. Renormalisation is NOT done here: `project_sphere_kernel`
    follows, unconditionally, as it does for every other `z` producer."""
    var t = Int(global_idx.x)
    if t >= NW * D:
        return
    var w = t // D
    var k = t - w * D
    var acc: Scalar[DT] = 0
    for j in range(SEQ):
        acc += src[unsafe_offset=(w * SEQ + j) * D + k]
    var m = acc / Scalar[DT](SEQ)
    for j in range(SEQ):
        dst[unsafe_offset=(w * SEQ + j) * D + k] = m


def expand_windows_kernel[NW: Int, SEQ: Int](
    starts: Pointer[Scalar[IDX_DT], MutAnyOrigin],
    rows_s: Pointer[Scalar[IDX_DT], MutAnyOrigin],
    rows_sn: Pointer[Scalar[IDX_DT], MutAnyOrigin],
):
    """`rows_s[w·SEQ + j] = starts[w] + j`, `rows_sn[...] = starts[w] + j + 1`.
    A window start is VALID only if `start + SEQ` is still inside its
    episode; the caller's start table carries that guarantee (built from the
    store's episode index), not this kernel."""
    var t = Int(global_idx.x)
    if t >= NW * SEQ:
        return
    var w = t // SEQ
    var j = t - w * SEQ
    var s = Int(starts[unsafe_offset=w])
    rows_s[unsafe_offset=t] = Scalar[IDX_DT](s + j)
    rows_sn[unsafe_offset=t] = Scalar[IDX_DT](s + j + 1)


def z_mixture3_kernel[D: Int, BATCH: Int](
    z: Pointer[Scalar[DT], MutAnyOrigin],
    gauss: Pointer[Scalar[DT], MutAnyOrigin],
    b_goal: Pointer[Scalar[DT], MutAnyOrigin],
    z_expert: Pointer[Scalar[DT], MutAnyOrigin],
    pick: Pointer[Scalar[DT], MutAnyOrigin],
    p_goal: Scalar[DT],
    p_expert: Scalar[DT],
    n_goal: Int32,
    n_expert: Int32,
):
    """BFM-Zero's training mixture (`sample_mixed_z`): a row is a GOAL
    encoding `B(s+)` with probability `p_goal`, an EXPERT trajectory
    encoding with probability `p_expert`, else Gaussian (uniform on the
    sphere after projection). `pick[2i]` chooses the branch, `pick[2i+1]`
    the source row — uniforms, drawn outside (see `uniform01_kernel`).
    `project_sphere_kernel` must follow."""
    var i = Int(global_idx.x)
    if i >= BATCH:
        return
    var base = i * D
    var u = pick[unsafe_offset=2 * i]
    var r = pick[unsafe_offset=2 * i + 1]
    var ng = Int(n_goal)
    var ne = Int(n_expert)
    if u < p_goal and ng > 0:
        var src = Int(r * Scalar[DT](ng))
        if src >= ng:
            src = ng - 1
        if src < 0:
            src = 0
        for k in range(D):
            z[unsafe_offset=base + k] = b_goal[unsafe_offset=src * D + k]
    elif u < p_goal + p_expert and ne > 0:
        var src = Int(r * Scalar[DT](ne))
        if src >= ne:
            src = ne - 1
        if src < 0:
            src = 0
        for k in range(D):
            z[unsafe_offset=base + k] = z_expert[unsafe_offset=src * D + k]
    else:
        for k in range(D):
            z[unsafe_offset=base + k] = gauss[unsafe_offset=base + k]


def lerp_rows_kernel[BATCH: Int, W: Int](
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    a: Pointer[Scalar[DT], MutAnyOrigin],
    b: Pointer[Scalar[DT], MutAnyOrigin],
    alpha: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`dst[i, :] = alpha[i]·a[i, :] + (1 − alpha[i])·b[i, :]` — the WGAN-GP
    interpolation between a real and a fake row, one weight per row."""
    var t = Int(global_idx.x)
    if t >= BATCH * W:
        return
    var i = t // W
    var al = alpha[unsafe_offset=i]
    dst[unsafe_offset=t] = al * a[unsafe_offset=t] + (Scalar[DT](1.0) - al) * b[unsafe_offset=t]


def clamp_kernel[N: Int](
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    src: Pointer[Scalar[DT], MutAnyOrigin],
    lo: Scalar[DT],
    hi: Scalar[DT],
):
    var t = Int(global_idx.x)
    if t >= N:
        return
    var v = src[unsafe_offset=t]
    if v < lo:
        v = lo
    elif v > hi:
        v = hi
    dst[unsafe_offset=t] = v


def axpy_by_mag_kernel[N: Int](
    y: Pointer[Scalar[DT], MutAnyOrigin],
    x: Pointer[Scalar[DT], MutAnyOrigin],
    mag: Pointer[Scalar[DT], MutAnyOrigin],
    base: Scalar[DT],
):
    """`y += base·|mag[0]|·x` — the inverse of `scale_by_inv_mag_kernel`.
    BFM-Zero's `scale_reg`: the CPR actor term is multiplied by the detached
    magnitude of the FB value term so `reg_coeff` is a RATIO."""
    var t = Int(global_idx.x)
    if t >= N:
        return
    var m = mag[unsafe_offset=0]
    if m < Scalar[DT](0):
        m = -m
    y[unsafe_offset=t] = y[unsafe_offset=t] + base * m * x[unsafe_offset=t]


def diff_scale_kernel[N: Int](
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    a: Pointer[Scalar[DT], MutAnyOrigin],
    b: Pointer[Scalar[DT], MutAnyOrigin],
    s: Scalar[DT],
):
    """`dst = s·(a − b)` — a TD residual scaled into a cotangent."""
    var t = Int(global_idx.x)
    if t >= N:
        return
    dst[unsafe_offset=t] = s * (a[unsafe_offset=t] - b[unsafe_offset=t])


# ── host twins ─────────────────────────────────────────────────────────


def fill_t[target: StaticString, N: Int](
    mut y: Tensor, v: Scalar[DT], ctx: Optional[DeviceContext] = None
) raises:
    ensure_t[target](y, N, ctx)
    comptime if target == "cpu":
        for i in range(N):
            y.data[i] = v
    else:
        var d = ctx.value()
        d.enqueue_function[fill_kernel[N]](
            y.dev.value().unsafe_ptr(), v,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def uniform01_dev_t[target: StaticString, N: Int](
    mut t: Tensor,
    seed: UInt64,
    ref [MutAnyOrigin] offset_buf: DeviceBuffer[DType.uint64],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`t[i] ~ U[0,1)`. GPU: Philox with a DEVICE offset, bumped in-sequence
    (capture-safe, see `gaussian_dev_t`). CPU: the host RNG."""
    ensure_t[target](t, N, ctx)
    comptime if target == "cpu":
        for i in range(N):
            t.data[i] = Scalar[DT](random_float64())
    else:
        var d = ctx.value()
        var off = LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin](
            mptr(offset_buf.unsafe_ptr())
        )
        d.enqueue_function[uniform01_dev_kernel[N]](
            t.dev.value().unsafe_ptr(), seed, off,
            grid_dim=_blocks(N), block_dim=TPB,
        )
        comptime AMT = 2 * N
        d.enqueue_function[advance_rng_offset_kernel[AMT]](
            off, grid_dim=1, block_dim=1
        )


def window_mean_t[target: StaticString, SEQ: Int, D: Int, NW: Int](
    mut dst: Tensor, mut src: Tensor, ctx: Optional[DeviceContext] = None
) raises:
    ensure_t[target](dst, NW * SEQ * D, ctx)
    comptime if target == "cpu":
        for w in range(NW):
            for k in range(D):
                var acc = Float64(0)
                for j in range(SEQ):
                    acc += Float64(src.data[(w * SEQ + j) * D + k])
                var m = Scalar[DT](acc / Float64(SEQ))
                for j in range(SEQ):
                    dst.data[(w * SEQ + j) * D + k] = m
    else:
        var d = ctx.value()
        d.enqueue_function[window_mean_kernel[SEQ, D, NW]](
            src.dev.value().unsafe_ptr(), dst.dev.value().unsafe_ptr(),
            grid_dim=_blocks(NW * D), block_dim=TPB,
        )


def project_sphere_t[target: StaticString, D: Int, BATCH: Int](
    mut z: Tensor, ctx: Optional[DeviceContext] = None
) raises:
    """Rows of `z` onto the radius-sqrt(D) sphere; same degenerate-row rule
    as `project_sphere_kernel` / `z_sampler._project_to_sphere`."""
    comptime if target == "cpu":
        var radius = sqrt(Float64(D))
        for i in range(BATCH):
            var acc = Float64(0)
            for k in range(D):
                var v = Float64(z.data[i * D + k])
                acc += v * v
            var n = sqrt(acc)
            if n < 1e-12:
                for k in range(D):
                    z.data[i * D + k] = Scalar[DT](0)
                z.data[i * D] = Scalar[DT](radius)
            else:
                var s = Scalar[DT](radius / n)
                for k in range(D):
                    z.data[i * D + k] = z.data[i * D + k] * s
    else:
        var d = ctx.value()
        d.enqueue_function[project_sphere_kernel[D, BATCH]](
            z.dev.value().unsafe_ptr(), Scalar[DT](sqrt(Float64(D))),
            grid_dim=_blocks(BATCH), block_dim=TPB,
        )


def lerp_rows_t[target: StaticString, BATCH: Int, W: Int](
    mut dst: Tensor, mut a: Tensor, mut b: Tensor, mut alpha: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    ensure_t[target](dst, BATCH * W, ctx)
    comptime if target == "cpu":
        for i in range(BATCH):
            var al = alpha.data[i]
            for k in range(W):
                var t = i * W + k
                dst.data[t] = al * a.data[t] + (Scalar[DT](1.0) - al) * b.data[t]
    else:
        var d = ctx.value()
        d.enqueue_function[lerp_rows_kernel[BATCH, W]](
            dst.dev.value().unsafe_ptr(), a.dev.value().unsafe_ptr(),
            b.dev.value().unsafe_ptr(), alpha.dev.value().unsafe_ptr(),
            grid_dim=_blocks(BATCH * W), block_dim=TPB,
        )


def clamp_t[target: StaticString, N: Int](
    mut dst: Tensor, mut src: Tensor, lo: Scalar[DT], hi: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    ensure_t[target](dst, N, ctx)
    comptime if target == "cpu":
        for i in range(N):
            var v = src.data[i]
            if v < lo:
                v = lo
            elif v > hi:
                v = hi
            dst.data[i] = v
    else:
        var d = ctx.value()
        d.enqueue_function[clamp_kernel[N]](
            dst.dev.value().unsafe_ptr(), src.dev.value().unsafe_ptr(), lo, hi,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def axpy_by_mag_t[target: StaticString, N: Int](
    mut y: Tensor, mut x: Tensor, mut mag: Tensor, base: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    comptime if target == "cpu":
        var m = abs(Float64(mag.data[0]))
        var a = Scalar[DT](Float64(base) * m)
        for i in range(N):
            y.data[i] = y.data[i] + a * x.data[i]
    else:
        var d = ctx.value()
        d.enqueue_function[axpy_by_mag_kernel[N]](
            y.dev.value().unsafe_ptr(), x.dev.value().unsafe_ptr(),
            mag.dev.value().unsafe_ptr(), base,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def diff_scale_t[target: StaticString, N: Int](
    mut dst: Tensor, mut a: Tensor, mut b: Tensor, s: Scalar[DT],
    ctx: Optional[DeviceContext] = None,
) raises:
    ensure_t[target](dst, N, ctx)
    comptime if target == "cpu":
        for i in range(N):
            dst.data[i] = s * (a.data[i] - b.data[i])
    else:
        var d = ctx.value()
        d.enqueue_function[diff_scale_kernel[N]](
            dst.dev.value().unsafe_ptr(), a.dev.value().unsafe_ptr(),
            b.dev.value().unsafe_ptr(), s,
            grid_dim=_blocks(N), block_dim=TPB,
        )


def sq_diff_mean_into_t[target: StaticString, N: Int](
    mut a: Tensor, mut b: Tensor, mut acc: Tensor,
    ctx: Optional[DeviceContext] = None,
) raises:
    """`acc[0] = mean((a − b)^2)`, no D2H (capture-safe)."""
    ensure_t[target](acc, 1, ctx)
    comptime if target == "cpu":
        var s = Float64(0)
        for i in range(N):
            var r = Float64(a.data[i]) - Float64(b.data[i])
            s += r * r
        acc.data[0] = Scalar[DT](s / Float64(N))
    else:
        var d = ctx.value()
        d.enqueue_function[sq_diff_reduce_kernel[N]](
            a.dev.value().unsafe_ptr(), b.dev.value().unsafe_ptr(),
            acc.dev.value().unsafe_ptr(),
            grid_dim=1, block_dim=TPB_REDUCE,
        )


def z_relabel3_kernel[D: Int, BATCH: Int](
    z: Pointer[Scalar[DT], MutAnyOrigin],
    gauss: Pointer[Scalar[DT], MutAnyOrigin],
    b_goal: Pointer[Scalar[DT], MutAnyOrigin],
    z_expert: Pointer[Scalar[DT], MutAnyOrigin],
    pick: Pointer[Scalar[DT], MutAnyOrigin],
    keep_frac: Scalar[DT],
    p_goal: Scalar[DT],
    p_expert: Scalar[DT],
    n_goal: Int32,
    n_expert: Int32,
):
    """The ONLINE relabel with BFM-Zero's three-way mixture: `z` arrives
    holding each row's STORED z; a row keeps it with probability
    `keep_frac`, else it is overwritten by `z_mixture3_kernel`'s draw
    (goal `B(s+)` / expert window encoding / Gaussian). Three uniforms per
    row in `pick` (keep, branch, source row). `project_sphere_kernel` must
    follow. `online.z_relabel_kernel` is the two-way (no expert) form."""
    var i = Int(global_idx.x)
    if i >= BATCH:
        return
    var base = i * D
    if pick[unsafe_offset=3 * i] < keep_frac:
        return
    var u = pick[unsafe_offset=3 * i + 1]
    var r = pick[unsafe_offset=3 * i + 2]
    var ng = Int(n_goal)
    var ne = Int(n_expert)
    if u < p_goal and ng > 0:
        var src = Int(r * Scalar[DT](ng))
        if src >= ng:
            src = ng - 1
        if src < 0:
            src = 0
        for k in range(D):
            z[unsafe_offset=base + k] = b_goal[unsafe_offset=src * D + k]
    elif u < p_goal + p_expert and ne > 0:
        var src = Int(r * Scalar[DT](ne))
        if src >= ne:
            src = ne - 1
        if src < 0:
            src = 0
        for k in range(D):
            z[unsafe_offset=base + k] = z_expert[unsafe_offset=src * D + k]
    else:
        for k in range(D):
            z[unsafe_offset=base + k] = gauss[unsafe_offset=base + k]
