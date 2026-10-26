"""BatchNorm2D[C, H, W, MOMENTUM, EPSILON] — per-channel BN for spatial inputs.

Transformed from legacy `nn.primitives.BatchNorm2D` (surface-only change). The
per-channel reduction over batch×spatial, the multi-block GPU reduction
(partial → finalize → scatter, the Σx/Σx² one-pass variance form), the finite-
guarded EMA, and the train/eval split are all carried over verbatim. Same State
treatment as BatchNorm1D: γ/β are Param (optimized); running_mean/var are
owned `Tensor`s evolved only by the forward EMA (not optimized).
"""

from std.math import sqrt
from max.gpu import thread_idx, block_idx, global_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from std.sys.info import default_accelerator
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile

from noeira.nn.constants import CPU_SIMD_W, DT, LAYOUT_NCHW, LAYOUT_NHWC
from ..core.polyak import polyak_tensor
from ..core.tensor import Tensor, TensorImpl
from .linear import _cast_f2b_kernel, _cast_b2f_kernel
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.state import State
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


comptime BN2D_DEFAULT_EPS: Float64 = 1e-5
comptime BN2D_DEFAULT_MOM: Float64 = 0.1
comptime BN2D_TPB: Int = 128
comptime BN2D_RBLOCKS: Int = 64
# NHWC coalesced reduction: row-chunk count (= partial-buffer rows). The NHWC
# reduction is transposed (thread-per-channel, coalesced over the inner C axis);
# CHUNKS sets its parallelism. 256 keeps the device busy while the partial scratch
# [C*CHUNKS] stays small.
comptime BN2D_NHWC_CHUNKS: Int = 256
# NHWC-2D occupancy fix (NVIDIA only). The 1-warp transposed reduction above is
# block_dim=C (1 warp, ~16k threads) → thread-starved vs the NCHW G-grouped path's
# ~262k. The 2D kernel uses block_dim = BN2D_NHWC_BLK = ROWS row-groups × C
# channels, each (rg,c) thread writing its OWN partial → 8 warps/block (C=32) =
# full occupancy, then a parallel block.sum finalize (grid=C). NVIDIA A/B (256
# chunks, the tuned sweet spot): NHWC-2D is 1.6-2x faster than the 1-warp and
# 0.98-1.50x vs the real NCHW BN (rep24/C4 at/under parity). Metal-gated OFF: the
# 2D launch ICEs the Metal backend in a large module, and Apple keeps the proven
# 1-warp kernel anyway — so the gate (default_accelerator) means the 2D is
# never codegen'd for Metal. Requires C | BN2D_NHWC_BLK (else falls back to 1-warp).
comptime BN2D_NHWC_BLK: Int = 256


@always_inline
def _bn_off[LAYOUT: Int, C: Int, SPATIAL: Int](c: Int, s: Int) -> Int:
    """Within-sample flat offset of (channel c, spatial s). NCHW c*SPATIAL+s
    (channel-outer, spatial contiguous) | NHWC s*C+c (channel-inner). The per-
    channel stats (mean/var/γ/β/running) are indexed by `c` alone, so this single
    offset is BN2D's ONLY layout-sensitive quantity. NCHW reproduces the prior
    formula exactly (bit-identical).

    NOTE (perf): the block-per-channel reduction/normalize kernels stay coalesced
    for NCHW (spatial contiguous) but become stride-C *uncoalesced* for NHWC. This
    offset-swap is correctness-first; the coalesced NHWC transposed reduction
    (the 3–5× win — see bench_bn_pool_nhwc_parity_gpu.mojo) is a follow-up that
    swaps only the NHWC reduction kernel, not this layout-agnostic structure."""
    comptime if LAYOUT == LAYOUT_NHWC:
        return s * C + c
    else:
        return c * SPATIAL + s


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _bn2d_partial_stats_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    FLAT: Int,
    G: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](
    input: Tile[DT, BATCH, FLAT],
    partial_sum: Tile[DT, C * G],
    partial_sumsq: Tile[DT, C * G],
):
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (BATCH + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > BATCH:
        b1 = BATCH
    var my_sum: input.ElementType = 0.0
    var my_sumsq: input.ElementType = 0.0
    for b in range(b0, b1):
        var s = t
        while s < SPATIAL:
            var x = input[b, _bn_off[LAYOUT, C, SPATIAL](c, s)]
            my_sum += x
            my_sumsq += x * x
            s += BN2D_TPB
    var bsum = block.sum[block_size=BN2D_TPB, broadcast=False](val=my_sum)
    var bsq = block.sum[block_size=BN2D_TPB, broadcast=False](val=my_sumsq)
    if t == 0:
        partial_sum[c * G + g] = bsum[0]
        partial_sumsq[c * G + g] = bsq[0]


def _bn2d_finalize_stats_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    G: Int,
    EPSILON: Float64,
](
    partial_sum: Tile[DT, C * G],
    partial_sumsq: Tile[DT, C * G],
    cache_mean: Tile[DT, C],
    cache_var: Tile[DT, C],
    cache_inv_std: Tile[DT, C],
):
    var c = Int(block_idx.x)
    if c >= C:
        return
    if Int(thread_idx.x) != 0:
        return
    var s: partial_sum.ElementType = 0.0
    var sq: partial_sumsq.ElementType = 0.0
    for g in range(G):
        s += partial_sum[c * G + g]
        sq += partial_sumsq[c * G + g]
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(BATCH * SPATIAL))
    var mean = s * inv_n
    var var_ = sq * inv_n - mean * mean
    if var_ < Scalar[DT](0.0):
        var_ = Scalar[DT](0.0)
    var eps: partial_sum.ElementType = Scalar[DT](EPSILON)
    var inv_std: partial_sum.ElementType = 1.0 / sqrt(var_ + eps)
    cache_mean[c] = mean
    cache_var[c] = var_
    cache_inv_std[c] = inv_std


def _bn2d_update_running_kernel[
    C: Int,
    MOMENTUM: Float64,
](
    cache_mean: Tile[DT, C],
    cache_var: Tile[DT, C],
    running_mean: Tile[DT, C],
    running_var: Tile[DT, C],
):
    """Running-stat EMA, in its OWN kernel (one thread / channel). Split out of
    `_bn2d_finalize_stats_kernel`: when that kernel ran the EMA inside its
    reduction body, the running_mean/running_var read-modify-write STORES were
    dropped by the NVIDIA backend at the B>64 (G=64) instantiation — running
    stats stayed pinned at init (0/1), so eval (running-stat) accuracy collapsed
    while train (batch-stat) accuracy was fine. As the kernel's sole job the
    stores survive. Unconditional (matches the CPU path; no NaN guard)."""
    var c = Int(block_idx.x)
    if c >= C:
        return
    var mom = Scalar[DT](MOMENTUM)
    var one_m = Scalar[DT](1.0) - mom
    running_mean[c] = one_m * running_mean[c] + mom * cache_mean[c]
    running_var[c] = one_m * running_var[c] + mom * cache_var[c]


def _bn2d_normalize_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    FLAT: Int,
    G: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](
    input: Tile[DT, BATCH, FLAT],
    output: Tile[DT, BATCH, FLAT],
    gamma: Tile[DT, C],
    beta: Tile[DT, C],
    cache_mean: Tile[DT, C],
    cache_inv_std: Tile[DT, C],
    cache_xhat: Tile[DT, BATCH, FLAT],
):
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (BATCH + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > BATCH:
        b1 = BATCH
    var mean = cache_mean[c]
    var inv_std = cache_inv_std[c]
    var gm = gamma[c]
    var bt = beta[c]
    for b in range(b0, b1):
        var s = t
        while s < SPATIAL:
            var off = _bn_off[LAYOUT, C, SPATIAL](c, s)
            var xh = (input[b, off] - mean) * inv_std
            cache_xhat[b, off] = xh
            output[b, off] = gm * xh + bt
            s += BN2D_TPB


def _bn2d_forward_eval_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    FLAT: Int,
    EPSILON: Float64,
    LAYOUT: Int = LAYOUT_NCHW,
](
    input: Tile[DT, BATCH, FLAT],
    output: Tile[DT, BATCH, FLAT],
    gamma: Tile[DT, C],
    beta: Tile[DT, C],
    running_mean: Tile[DT, C],
    running_var: Tile[DT, C],
):
    var c = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if c >= C:
        return
    var eps = Scalar[DT](EPSILON)
    var rm = running_mean[c]
    var rv = running_var[c]
    var inv_std: input.ElementType = 1.0 / sqrt(rv + eps)
    var g = gamma[c]
    var bt = beta[c]
    for b in range(BATCH):
        var s = t
        while s < SPATIAL:
            var off = _bn_off[LAYOUT, C, SPATIAL](c, s)
            var x = input[b, off]
            output[b, off] = g * (x - rm) * inv_std + bt
            s += BN2D_TPB


def _bn2d_eval_bwd_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    FLAT: Int,
    EPSILON: Float64,
    LAYOUT: Int = LAYOUT_NCHW,
](
    grad_output: Tile[DT, BATCH, FLAT],
    gamma: Tile[DT, C],
    running_var: Tile[DT, C],
    grad_input: Tile[DT, BATCH, FLAT],
):
    # Eval-mode input gradient: running stats are CONSTANTS, so there is no batch
    # coupling — gi = γ·inv_std·dy, inv_std = 1/√(running_var+ε). One block per
    # channel (mirrors the eval forward kernel); γ/β grads are not needed (this
    # path is only used for a FROZEN backbone, e.g. the perceptual loss).
    var c = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if c >= C:
        return
    var eps = Scalar[DT](EPSILON)
    var inv_std: grad_output.ElementType = 1.0 / sqrt(running_var[c] + eps)
    var g = gamma[c]
    for b in range(BATCH):
        var s = t
        while s < SPATIAL:
            var off = _bn_off[LAYOUT, C, SPATIAL](c, s)
            grad_input[b, off] = g * inv_std * grad_output[b, off]
            s += BN2D_TPB


def _bn2d_eval_bwd_affine_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    FLAT: Int,
    EPSILON: Float64,
    LAYOUT: Int = LAYOUT_NCHW,
](
    grad_output: Tile[DT, BATCH, FLAT],
    input: Tile[DT, BATCH, FLAT],
    gamma: Tile[DT, C],
    running_mean: Tile[DT, C],
    running_var: Tile[DT, C],
    grad_input: Tile[DT, BATCH, FLAT],
    grad_gamma: Tile[DT, C],
    grad_beta: Tile[DT, C],
):
    """Eval-mode backward INCLUDING the affine gradients.

    Running stats are constants, so there is no batch coupling in the input
    gradient — but γ and β still receive gradient, because the loss depends on
    them:

        inv_std   = 1/sqrt(running_var + eps)
        xhat      = (x - running_mean) * inv_std
        gi        = gamma * inv_std * dy
        dgamma[c] = sum_over(b,s) dy * xhat
        dbeta[c]  = sum_over(b,s) dy

    `xhat` is RECOMPUTED from the input rather than read from a cache: the eval
    forward does not populate `cache_xhat` on GPU, and making it do so would put
    a `[BATCH, FLAT]` write on every inference forward — a real cost on a path
    (AlphaZero/MuZero selfplay) that never backwards at all. Recomputing here
    keeps the cost on the rare backward.

    One block per channel, threads striding the batch*spatial extent, then a
    block reduction — the same shape as the eval forward kernel.
    """
    var c = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if c >= C:
        return
    var eps = Scalar[DT](EPSILON)
    var inv_std: grad_output.ElementType = 1.0 / sqrt(running_var[c] + eps)
    var mean = running_mean[c]
    var g = gamma[c]
    var acc_dg: Scalar[DT] = 0.0
    var acc_db: Scalar[DT] = 0.0
    for b in range(BATCH):
        var s = t
        while s < SPATIAL:
            var off = _bn_off[LAYOUT, C, SPATIAL](c, s)
            var dy = rebind[Scalar[DT]](grad_output[b, off])
            var xh = (
                rebind[Scalar[DT]](input[b, off]) - rebind[Scalar[DT]](mean)
            ) * rebind[Scalar[DT]](inv_std)
            grad_input[b, off] = g * inv_std * grad_output[b, off]
            acc_dg += dy * xh
            acc_db += dy
            s += BN2D_TPB
    var sg = block.sum[block_size=BN2D_TPB, broadcast=False](val=acc_dg)
    var sb = block.sum[block_size=BN2D_TPB, broadcast=False](val=acc_db)
    if t == 0:
        # ACCUMULATE (+=), matching the CPU path and every other param grad in
        # the framework — a graph may reach one module through several edges.
        grad_gamma[c] = rebind[Scalar[DT]](grad_gamma[c]) + sg[0]
        grad_beta[c] = rebind[Scalar[DT]](grad_beta[c]) + sb[0]


def _bn2d_bwd_partial_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    FLAT: Int,
    G: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](
    grad_output: Tile[DT, BATCH, FLAT],
    gamma: Tile[DT, C],
    cache_xhat: Tile[DT, BATCH, FLAT],
    p_dxhat: Tile[DT, C * G],
    p_dxhat_xhat: Tile[DT, C * G],
    p_dgamma: Tile[DT, C * G],
    p_dbeta: Tile[DT, C * G],
):
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (BATCH + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > BATCH:
        b1 = BATCH
    var gm = gamma[c]
    var s_dxhat: grad_output.ElementType = 0.0
    var s_dxx: grad_output.ElementType = 0.0
    var s_dg: grad_output.ElementType = 0.0
    var s_db: grad_output.ElementType = 0.0
    for b in range(b0, b1):
        var s = t
        while s < SPATIAL:
            var off = _bn_off[LAYOUT, C, SPATIAL](c, s)
            var dy = grad_output[b, off]
            var xh = cache_xhat[b, off]
            var dxhat = dy * gm
            s_dxhat += dxhat
            s_dxx += dxhat * xh
            s_dg += dy * xh
            s_db += dy
            s += BN2D_TPB
    var a = block.sum[block_size=BN2D_TPB, broadcast=False](val=s_dxhat)
    var bb = block.sum[block_size=BN2D_TPB, broadcast=False](val=s_dxx)
    var cc = block.sum[block_size=BN2D_TPB, broadcast=False](val=s_dg)
    var dd = block.sum[block_size=BN2D_TPB, broadcast=False](val=s_db)
    if t == 0:
        p_dxhat[c * G + g] = a[0]
        p_dxhat_xhat[c * G + g] = bb[0]
        p_dgamma[c * G + g] = cc[0]
        p_dbeta[c * G + g] = dd[0]


def _bn2d_bwd_finalize_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    G: Int,
    mode: StaticString,
](
    p_dxhat: Tile[DT, C * G],
    p_dxhat_xhat: Tile[DT, C * G],
    p_dgamma: Tile[DT, C * G],
    p_dbeta: Tile[DT, C * G],
    m1_out: Tile[DT, C],
    m2_out: Tile[DT, C],
    grad_gamma: Tile[DT, C],
    grad_beta: Tile[DT, C],
):
    var c = Int(block_idx.x)
    if c >= C:
        return
    if Int(thread_idx.x) != 0:
        return
    var sa: p_dxhat.ElementType = 0.0
    var sb: p_dxhat_xhat.ElementType = 0.0
    var sg: p_dgamma.ElementType = 0.0
    var sd: p_dbeta.ElementType = 0.0
    for g in range(G):
        sa += p_dxhat[c * G + g]
        sb += p_dxhat_xhat[c * G + g]
        sg += p_dgamma[c * G + g]
        sd += p_dbeta[c * G + g]
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](BATCH * SPATIAL)
    m1_out[c] = sa * inv_n
    m2_out[c] = sb * inv_n
    comptime if mode == "all":
        grad_gamma[c] += sg
        grad_beta[c] += sd


def _bn2d_bwd_scatter_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    FLAT: Int,
    G: Int,
    LAYOUT: Int = LAYOUT_NCHW,
](
    grad_output: Tile[DT, BATCH, FLAT],
    gamma: Tile[DT, C],
    cache_xhat: Tile[DT, BATCH, FLAT],
    cache_inv_std: Tile[DT, C],
    m1: Tile[DT, C],
    m2: Tile[DT, C],
    grad_input: Tile[DT, BATCH, FLAT],
):
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (BATCH + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > BATCH:
        b1 = BATCH
    var gm = gamma[c]
    var inv_std = cache_inv_std[c]
    var mm1 = m1[c]
    var mm2 = m2[c]
    for b in range(b0, b1):
        var s = t
        while s < SPATIAL:
            var off = _bn_off[LAYOUT, C, SPATIAL](c, s)
            var dy = grad_output[b, off]
            var xh = cache_xhat[b, off]
            var dxhat = dy * gm
            grad_input[b, off] = inv_std * (dxhat - mm1 - xh * mm2)
            s += BN2D_TPB


# ── NHWC coalesced kernels (channels-last; the perf path for LAYOUT=NHWC) ────
# The block-per-channel NCHW kernels above coalesce on NCHW (spatial contiguous)
# but are stride-C UNCOALESCED on NHWC. These transpose the reduction: a "row"
# r=(b*SP+s) is a contiguous C-vector at flat[r*C..], so thread-per-channel /
# loop-rows makes consecutive threads read consecutive channels = coalesced
# (cuDNN NHWC-BN pattern). Forward/backward stats reduce over CHUNKS row-chunks
# then finalize per channel; normalize/scatter are flat thread-per-element. The
# per-channel caches (mean/var/inv_std/γ/β/running) are layout-agnostic, so the
# EMA kernel (_bn2d_update_running_kernel) is reused unchanged.
def _bn2d_nhwc_partial_stats[
    C: Int, R: Int, CHUNKS: Int,
](
    input: Tile[DT, R * C],
    partial_sum: Tile[DT, CHUNKS * C],
    partial_sumsq: Tile[DT, CHUNKS * C],
):
    var chunk = Int(block_idx.x)
    var c = Int(thread_idx.x)
    if c >= C:
        return
    comptime rpc = (R + CHUNKS - 1) // CHUNKS
    var r0 = chunk * rpc
    if r0 >= R:
        partial_sum[chunk * C + c] = Scalar[DT](0.0)
        partial_sumsq[chunk * C + c] = Scalar[DT](0.0)
        return
    var r1 = r0 + rpc
    if r1 > R:
        r1 = R
    var s: Scalar[DT] = 0.0
    var sq: Scalar[DT] = 0.0
    for r in range(r0, r1):
        var x = rebind[Scalar[DT]](input[r * C + c])
        s += x
        sq += x * x
    partial_sum[chunk * C + c] = s
    partial_sumsq[chunk * C + c] = sq


def _bn2d_nhwc_finalize_stats[
    C: Int, RTOT: Int, CHUNKS: Int, EPSILON: Float64,
](
    partial_sum: Tile[DT, CHUNKS * C],
    partial_sumsq: Tile[DT, CHUNKS * C],
    cache_mean: Tile[DT, C],
    cache_var: Tile[DT, C],
    cache_inv_std: Tile[DT, C],
):
    var c = Int(global_idx.x)
    if c >= C:
        return
    var s: Scalar[DT] = 0.0
    var sq: Scalar[DT] = 0.0
    for k in range(CHUNKS):
        s += rebind[Scalar[DT]](partial_sum[k * C + c])
        sq += rebind[Scalar[DT]](partial_sumsq[k * C + c])
    var inv_n = Scalar[DT](1.0) / Scalar[DT](Float32(RTOT))
    var mean = s * inv_n
    var var_ = sq * inv_n - mean * mean
    if var_ < Scalar[DT](0.0):
        var_ = Scalar[DT](0.0)
    cache_mean[c] = mean
    cache_var[c] = var_
    cache_inv_std[c] = Scalar[DT](1.0) / sqrt(var_ + Scalar[DT](EPSILON))


# ── NHWC-2D forward stats (NVIDIA occupancy fix) — see BN2D_NHWC_BLK note ──────
# block_dim = BN2D_NHWC_BLK = ROWS row-groups × C channels (lane = rg*C + c). Each
# row-group strides its own row subset; thread (rg,c) writes its OWN partial at
# (chunk*ROWS+rg)*C+c (no shared mem). Partial buffer = CHUNKS*ROWS*C =
# CHUNKS*BN2D_NHWC_BLK. The cross-row-group reduction is the parallel finalize.
def _bn2d_nhwc_partial_stats_2d[
    C: Int, R: Int, CHUNKS: Int,
](
    input: Tile[DT, R * C],
    partial_sum: Tile[DT, CHUNKS * BN2D_NHWC_BLK],
    partial_sumsq: Tile[DT, CHUNKS * BN2D_NHWC_BLK],
):
    comptime ROWS = BN2D_NHWC_BLK // C
    var chunk = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    var c = lane % C
    var rg = lane // C
    comptime rpc = (R + CHUNKS - 1) // CHUNKS
    var r0 = chunk * rpc
    var r1 = r0 + rpc
    if r1 > R:
        r1 = R
    var s: Scalar[DT] = 0.0
    var sq: Scalar[DT] = 0.0
    var r = r0 + rg
    while r < r1:
        var x = rebind[Scalar[DT]](input[r * C + c])
        s += x
        sq += x * x
        r += ROWS
    var pidx = (chunk * ROWS + rg) * C + c
    partial_sum[pidx] = s
    partial_sumsq[pidx] = sq


# Parallel finalize: grid=C, one block per channel reduces its P = CHUNKS*ROWS
# partials (strided by C) via block.sum. Writes the same mean/var/inv_std caches.
def _bn2d_nhwc_finalize_stats_2d[
    C: Int, RTOT: Int, P: Int, EPSILON: Float64,
](
    partial_sum: Tile[DT, P * C],
    partial_sumsq: Tile[DT, P * C],
    cache_mean: Tile[DT, C],
    cache_var: Tile[DT, C],
    cache_inv_std: Tile[DT, C],
):
    var c = Int(block_idx.x)
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var s: Scalar[DT] = 0.0
    var sq: Scalar[DT] = 0.0
    var k = t
    while k < P:
        s += rebind[Scalar[DT]](partial_sum[k * C + c])
        sq += rebind[Scalar[DT]](partial_sumsq[k * C + c])
        k += BN2D_TPB
    var bs = block.sum[block_size=BN2D_TPB, broadcast=False](val=s)
    var bsq = block.sum[block_size=BN2D_TPB, broadcast=False](val=sq)
    if t == 0:
        var inv_n = Scalar[DT](1.0) / Scalar[DT](Float32(RTOT))
        var mean = bs[0] * inv_n
        var var_ = bsq[0] * inv_n - mean * mean
        if var_ < Scalar[DT](0.0):
            var_ = Scalar[DT](0.0)
        cache_mean[c] = mean
        cache_var[c] = var_
        cache_inv_std[c] = Scalar[DT](1.0) / sqrt(var_ + Scalar[DT](EPSILON))


def _bn2d_nhwc_normalize[
    C: Int, RC: Int,
](
    input: Tile[DT, RC],
    output: Tile[DT, RC],
    gamma: Tile[DT, C],
    beta: Tile[DT, C],
    cache_mean: Tile[DT, C],
    cache_inv_std: Tile[DT, C],
    cache_xhat: Tile[DT, RC],
):
    var idx = Int(global_idx.x)
    if idx >= RC:
        return
    var c = idx % C
    var xh = (
        rebind[Scalar[DT]](input[idx]) - rebind[Scalar[DT]](cache_mean[c])
    ) * rebind[Scalar[DT]](cache_inv_std[c])
    cache_xhat[idx] = xh
    output[idx] = rebind[Scalar[DT]](gamma[c]) * xh + rebind[Scalar[DT]](beta[c])


def _bn2d_nhwc_bwd_partial[
    C: Int, R: Int, CHUNKS: Int,
](
    grad_output: Tile[DT, R * C],
    gamma: Tile[DT, C],
    cache_xhat: Tile[DT, R * C],
    p_dxhat: Tile[DT, CHUNKS * C],
    p_dxhat_xhat: Tile[DT, CHUNKS * C],
    p_dgamma: Tile[DT, CHUNKS * C],
    p_dbeta: Tile[DT, CHUNKS * C],
):
    var chunk = Int(block_idx.x)
    var c = Int(thread_idx.x)
    if c >= C:
        return
    comptime rpc = (R + CHUNKS - 1) // CHUNKS
    var r0 = chunk * rpc
    if r0 >= R:
        p_dxhat[chunk * C + c] = Scalar[DT](0.0)
        p_dxhat_xhat[chunk * C + c] = Scalar[DT](0.0)
        p_dgamma[chunk * C + c] = Scalar[DT](0.0)
        p_dbeta[chunk * C + c] = Scalar[DT](0.0)
        return
    var r1 = r0 + rpc
    if r1 > R:
        r1 = R
    var gm = rebind[Scalar[DT]](gamma[c])
    var s_dxhat: Scalar[DT] = 0.0
    var s_dxx: Scalar[DT] = 0.0
    var s_dg: Scalar[DT] = 0.0
    var s_db: Scalar[DT] = 0.0
    for r in range(r0, r1):
        var dy = rebind[Scalar[DT]](grad_output[r * C + c])
        var xh = rebind[Scalar[DT]](cache_xhat[r * C + c])
        var dxhat = dy * gm
        s_dxhat += dxhat
        s_dxx += dxhat * xh
        s_dg += dy * xh
        s_db += dy
    p_dxhat[chunk * C + c] = s_dxhat
    p_dxhat_xhat[chunk * C + c] = s_dxx
    p_dgamma[chunk * C + c] = s_dg
    p_dbeta[chunk * C + c] = s_db


def _bn2d_nhwc_bwd_finalize[
    C: Int, RTOT: Int, CHUNKS: Int, mode: StaticString,
](
    p_dxhat: Tile[DT, CHUNKS * C],
    p_dxhat_xhat: Tile[DT, CHUNKS * C],
    p_dgamma: Tile[DT, CHUNKS * C],
    p_dbeta: Tile[DT, CHUNKS * C],
    m1_out: Tile[DT, C],
    m2_out: Tile[DT, C],
    grad_gamma: Tile[DT, C],
    grad_beta: Tile[DT, C],
):
    var c = Int(global_idx.x)
    if c >= C:
        return
    var sa: Scalar[DT] = 0.0
    var sb: Scalar[DT] = 0.0
    var sg: Scalar[DT] = 0.0
    var sd: Scalar[DT] = 0.0
    for k in range(CHUNKS):
        sa += rebind[Scalar[DT]](p_dxhat[k * C + c])
        sb += rebind[Scalar[DT]](p_dxhat_xhat[k * C + c])
        sg += rebind[Scalar[DT]](p_dgamma[k * C + c])
        sd += rebind[Scalar[DT]](p_dbeta[k * C + c])
    var inv_n = Scalar[DT](1.0) / Scalar[DT](Float32(RTOT))
    m1_out[c] = sa * inv_n
    m2_out[c] = sb * inv_n
    comptime if mode == "all":
        grad_gamma[c] = rebind[Scalar[DT]](grad_gamma[c]) + sg
        grad_beta[c] = rebind[Scalar[DT]](grad_beta[c]) + sd


# ── NHWC-2D backward stats (NVIDIA occupancy fix), mirrors the forward 2D pair ──
# block_dim = BN2D_NHWC_BLK = ROWS×C; thread (rg,c) reduces its strided row subset
# of the 4 reductions (dxhat, dxhat·xhat, dgamma, dbeta) and writes per-(rg,c)
# partials at (chunk*ROWS+rg)*C+c. Partial buffers = CHUNKS*BN2D_NHWC_BLK each.
def _bn2d_nhwc_bwd_partial_2d[
    C: Int, R: Int, CHUNKS: Int,
](
    grad_output: Tile[DT, R * C],
    gamma: Tile[DT, C],
    cache_xhat: Tile[DT, R * C],
    p_dxhat: Tile[DT, CHUNKS * BN2D_NHWC_BLK],
    p_dxhat_xhat: Tile[DT, CHUNKS * BN2D_NHWC_BLK],
    p_dgamma: Tile[DT, CHUNKS * BN2D_NHWC_BLK],
    p_dbeta: Tile[DT, CHUNKS * BN2D_NHWC_BLK],
):
    comptime ROWS = BN2D_NHWC_BLK // C
    var chunk = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    var c = lane % C
    var rg = lane // C
    comptime rpc = (R + CHUNKS - 1) // CHUNKS
    var r0 = chunk * rpc
    var r1 = r0 + rpc
    if r1 > R:
        r1 = R
    var gm = rebind[Scalar[DT]](gamma[c])
    var s_dxhat: Scalar[DT] = 0.0
    var s_dxx: Scalar[DT] = 0.0
    var s_dg: Scalar[DT] = 0.0
    var s_db: Scalar[DT] = 0.0
    var r = r0 + rg
    while r < r1:
        var dy = rebind[Scalar[DT]](grad_output[r * C + c])
        var xh = rebind[Scalar[DT]](cache_xhat[r * C + c])
        var dxhat = dy * gm
        s_dxhat += dxhat
        s_dxx += dxhat * xh
        s_dg += dy * xh
        s_db += dy
        r += ROWS
    var pidx = (chunk * ROWS + rg) * C + c
    p_dxhat[pidx] = s_dxhat
    p_dxhat_xhat[pidx] = s_dxx
    p_dgamma[pidx] = s_dg
    p_dbeta[pidx] = s_db


# Parallel backward finalize: grid=C, block reduces its P = CHUNKS*ROWS partials
# of all 4 sums via block.sum. Writes m1/m2; accumulates grad_γ/grad_β (mode=all).
def _bn2d_nhwc_bwd_finalize_2d[
    C: Int, RTOT: Int, P: Int, mode: StaticString,
](
    p_dxhat: Tile[DT, P * C],
    p_dxhat_xhat: Tile[DT, P * C],
    p_dgamma: Tile[DT, P * C],
    p_dbeta: Tile[DT, P * C],
    m1_out: Tile[DT, C],
    m2_out: Tile[DT, C],
    grad_gamma: Tile[DT, C],
    grad_beta: Tile[DT, C],
):
    var c = Int(block_idx.x)
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var sa: Scalar[DT] = 0.0
    var sb: Scalar[DT] = 0.0
    var sg: Scalar[DT] = 0.0
    var sd: Scalar[DT] = 0.0
    var k = t
    while k < P:
        sa += rebind[Scalar[DT]](p_dxhat[k * C + c])
        sb += rebind[Scalar[DT]](p_dxhat_xhat[k * C + c])
        sg += rebind[Scalar[DT]](p_dgamma[k * C + c])
        sd += rebind[Scalar[DT]](p_dbeta[k * C + c])
        k += BN2D_TPB
    var bsa = block.sum[block_size=BN2D_TPB, broadcast=False](val=sa)
    var bsb = block.sum[block_size=BN2D_TPB, broadcast=False](val=sb)
    var bsg = block.sum[block_size=BN2D_TPB, broadcast=False](val=sg)
    var bsd = block.sum[block_size=BN2D_TPB, broadcast=False](val=sd)
    if t == 0:
        var inv_n = Scalar[DT](1.0) / Scalar[DT](Float32(RTOT))
        m1_out[c] = bsa[0] * inv_n
        m2_out[c] = bsb[0] * inv_n
        comptime if mode == "all":
            grad_gamma[c] = rebind[Scalar[DT]](grad_gamma[c]) + bsg[0]
            grad_beta[c] = rebind[Scalar[DT]](grad_beta[c]) + bsd[0]


def _bn2d_nhwc_bwd_scatter[
    C: Int, RC: Int,
](
    grad_output: Tile[DT, RC],
    gamma: Tile[DT, C],
    cache_xhat: Tile[DT, RC],
    cache_inv_std: Tile[DT, C],
    m1: Tile[DT, C],
    m2: Tile[DT, C],
    grad_input: Tile[DT, RC],
):
    var idx = Int(global_idx.x)
    if idx >= RC:
        return
    var c = idx % C
    var dy = rebind[Scalar[DT]](grad_output[idx])
    var xh = rebind[Scalar[DT]](cache_xhat[idx])
    var dxhat = dy * rebind[Scalar[DT]](gamma[c])
    grad_input[idx] = rebind[Scalar[DT]](cache_inv_std[c]) * (
        dxhat - rebind[Scalar[DT]](m1[c]) - xh * rebind[Scalar[DT]](m2[c])
    )


struct BatchNorm2D[
    C_: Int,
    H_: Int,
    W_: Int,
    MOMENTUM: Float64 = BN2D_DEFAULT_MOM,
    EPSILON: Float64 = BN2D_DEFAULT_EPS,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
](Module):
    comptime ARITY = 1
    comptime FLAT_DIM = Self.C_ * Self.H_ * Self.W_
    comptime IN_DIMS = Array[Int, 1](fill=Self.FLAT_DIM)
    comptime OUT_DIM = Self.FLAT_DIM
    comptime SPATIAL = Self.H_ * Self.W_
    # NHWC-2D occupancy path is on iff NVIDIA + channels-last + C divides the 2D
    # block (else the 1-warp transposed kernel — the Apple/Metal path — is used).
    comptime USE_2D_NHWC = (
        Self.LAYOUT == LAYOUT_NHWC
        and default_accelerator().is_nvidia_gpu()
        and BN2D_NHWC_BLK % Self.C_ == 0
    )
    # Activation-flow dtype (AMP §3 fp32-INTERNAL): BN accepts/emits ACT_DT but
    # computes stats/normalize in fp32 internally. ACT_DT == DT (default) →
    # the cast wrappers collapse and the fp32 path is byte-identical.
    comptime ACT_DT = Self.ADT

    var gamma: Param["gamma", False, Self.C_]
    var beta: Param["beta", False, Self.C_]
    var running_mean: State["running_mean", Self.C_]  # [C] State
    var running_var: State["running_var", Self.C_]  # [C] State
    var cache_xhat: Tensor  # [BATCH, FLAT]
    var cache_inv_std: Tensor  # [C]
    var cache_mean: Tensor  # [C] (GPU multiblock normalize)
    var cache_var: Tensor  # [C] (GPU: batch var, fed to the running-stat EMA)
    # Multi-block reduction scratch ([C·RBLOCKS] or [C]).
    var bn_psum: Tensor
    var bn_psumsq: Tensor
    var bn_pdg: Tensor
    var bn_pdb: Tensor
    var bn_m1: Tensor
    var bn_m2: Tensor
    var cache_is_training: Bool
    var training: Bool
    var frozen: Bool
    """`FrozenBatchNorm2d`: eval-mode ALWAYS, and gamma/beta take no gradient.

    ⚠ NOT the same as `training = False`. Eval mode still trains gamma/beta —
    it only stops the batch statistics being used and the running statistics
    being updated. DETR's `FrozenBatchNorm2d` (`references/act-main/detr/models/
    backbone.py:21`, and LeRobot imports torchvision's) registers weight, bias,
    running_mean and running_var as BUFFERS, so all four are constants. Both
    ACT implementations build their ResNet backbone with it.

    That matters exactly when the four are worth keeping: with pretrained
    weights loaded, training-mode BatchNorm EMAs the ImageNet statistics away
    at momentum 0.1 — 2.7e-05 of the original after 100 steps — while never
    reading them, because training-mode forward uses batch statistics. Freezing
    is what makes ImageNet statistics mean anything downstream."""

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.C_]()
        self.beta = Param["beta", False, Self.C_]()
        self.running_mean = State["running_mean", Self.C_]()
        self.running_var = State["running_var", Self.C_]()
        self.cache_xhat = Tensor()
        self.cache_inv_std = Tensor()
        self.cache_mean = Tensor()
        self.cache_var = Tensor()
        self.bn_psum = Tensor()
        self.bn_psumsq = Tensor()
        self.bn_pdg = Tensor()
        self.bn_pdb = Tensor()
        self.bn_m1 = Tensor()
        self.bn_m2 = Tensor()
        self.cache_is_training = False
        self.training = True
        self.frozen = False

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var bn = Self()
        bn.gamma = Param["gamma", False, Self.C_].make[target](ctx)
        bn.beta = Param["beta", False, Self.C_].make[target](ctx)
        for k in range(Self.C_):
            bn.gamma.val.data[k] = Scalar[DT](1.0)
        bn.running_mean = State["running_mean", Self.C_].make[target](ctx)
        bn.running_var = State["running_var", Self.C_].make[target](ctx)
        for k in range(Self.C_):
            bn.running_var.t.data[k] = Scalar[DT](1.0)
        comptime if target != "cpu":
            var c = ctx.value()
            bn.gamma.val.upload(c)
            bn.beta.val.upload(c)
            bn.running_mean.t.upload(c)
            bn.running_var.t.upload(c)
            # Multi-block scratch + channel caches. NCHW partials are [C*RBLOCKS]
            # (channel-major, G batch-groups); NHWC partials are [C*CHUNKS]
            # (chunk-major, transposed reduction) — size to the larger so one
            # alloc serves both layouts.
            # NHWC-2D partials are [CHUNKS*ROWS*C = CHUNKS*BN2D_NHWC_BLK] (per
            # row-group); 1-warp NHWC are [C*CHUNKS]; NCHW are [C*RBLOCKS]. Size
            # to whichever path is active so one alloc serves it (and the 4 bwd
            # partial buffers reuse the same size).
            comptime PR = (
                BN2D_NHWC_CHUNKS * BN2D_NHWC_BLK if Self.USE_2D_NHWC
                else Self.C_ * (
                    BN2D_NHWC_CHUNKS if Self.LAYOUT
                    == LAYOUT_NHWC else BN2D_RBLOCKS
                )
            )
            bn.cache_inv_std.ensure_gpu(c, Self.C_)
            bn.cache_mean.ensure_gpu(c, Self.C_)
            bn.cache_var.ensure_gpu(c, Self.C_)
            bn.bn_psum.ensure_gpu(c, PR)
            bn.bn_psumsq.ensure_gpu(c, PR)
            bn.bn_pdg.ensure_gpu(c, PR)
            bn.bn_pdb.ensure_gpu(c, PR)
            bn.bn_m1.ensure_gpu(c, Self.C_)
            bn.bn_m2.ensure_gpu(c, Self.C_)
        return bn^

    def set_training(mut self, v: Bool):
        self.training = v

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Named-attr hook so a parent `Sequential`/`Repeat`/… can toggle BN
        train/eval via `net.set_attr["training"](1.0/0.0)` (the AZ CNN/ResNet
        drivers' BN switch). `value != 0` → training (batch stats + running-stat
        updates); else eval (running stats).

        `set_attr["frozen"](1.0)` additionally pins gamma/beta — see `frozen`.
        It OVERRIDES `training`, so the two can be set in either order and a
        later `train_mode(True)` cannot silently unfreeze a backbone."""
        comptime if ATTR == "training":
            self.training = value != Scalar[DT](0.0)
        comptime if ATTR == "frozen":
            self.frozen = value != Scalar[DT](0.0)

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        """Soft-update toward `src` (target <- online). The `Module`
        default is a NO-OP, so without this override a target net containing
        this layer would keep its init values forever while the online copy
        trains (the `LayerNorm` / `Conv2D` bug class; audit 2026-10-07)."""
        polyak_tensor[target, Self.C_](self.gamma.val, src.gamma.val, tau, ctx)
        polyak_tensor[target, Self.C_](self.beta.val, src.beta.val, tau, ctx)
        # The running statistics follow too: with tau = 1 this is the hard sync
        # `hard_copy` does (params AND State).
        polyak_tensor[target, Self.C_](self.running_mean.t, src.running_mean.t, tau, ctx)
        polyak_tensor[target, Self.C_](self.running_var.t, src.running_var.t, tau, ctx)


    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # AMP §3 fp32-internal: ACT_DT==DT → bit-identical fp32 path; else cast
        # the bf16 activation in→fp32, run the fp32 BN, cast out→bf16.
        comptime if Self.ACT_DT == DT:
            ref in0d = rebind[Tensor](inputs[0])
            ref outd = rebind[Tensor](out)
            self._forward_f32[target, B](in0d, outd, ctx)
        else:
            comptime N = B * Self.FLAT_DIM
            # LOCAL fp32 scratch (not self-fields → no mut-self aliasing).
            var in_f32 = Tensor()
            in_f32.ensure[target](N, ctx)
            var out_f32 = Tensor()
            out_f32.ensure[target](N, ctx)
            out.ensure[target](N, ctx)
            ref in0 = inputs[0]
            comptime if target == "cpu":
                for i in range(N):
                    in_f32.data[i] = in0.data[i].cast[DT]()
            else:
                var c = ctx.value()
                c.enqueue_function[_cast_b2f_kernel](
                    in0.dev.value(),
                    in_f32.dev.value(),
                    Int64(N),
                    grid_dim=(N + 255) // 256,
                    block_dim=256,
                )
            self._forward_f32[target, B](in_f32, out_f32, ctx)
            comptime if target == "cpu":
                for i in range(N):
                    out.data[i] = out_f32.data[i].cast[Self.ACT_DT]()
            else:
                var c = ctx.value()
                c.enqueue_function[_cast_f2b_kernel](
                    out_f32.dev.value(),
                    out.dev.value(),
                    Int64(N),
                    grid_dim=(N + 255) // 256,
                    block_dim=256,
                )

    def _forward_f32[target: StaticString, B: Int](
        mut self,
        mut in0: Tensor,
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        var eps = Scalar[DT](Self.EPSILON)
        comptime if target == "cpu":
            out.ensure(B * Self.FLAT_DIM)
            var in_p = in0.data.unsafe_ptr()
            var out_p = out.data.unsafe_ptr()
            var g_p = self.gamma.val.data.unsafe_ptr()
            var b_p = self.beta.val.data.unsafe_ptr()
            var rm_v = TileTensor(self.running_mean.t.data, row_major[Self.C_]())
            var rv_v = TileTensor(self.running_var.t.data, row_major[Self.C_]())
            var inv_n = Scalar[DT](1.0) / Scalar[DT](Float64(B * Self.SPATIAL))
            if self.training and not self.frozen:
                self.cache_xhat.ensure(B * Self.FLAT_DIM)
                self.cache_inv_std.ensure(Self.C_)
                var xhat_p = self.cache_xhat.data.unsafe_ptr()
                var inv_v = TileTensor(
                    self.cache_inv_std.data, row_major[Self.C_]()
                )
                var mom = Scalar[DT](Self.MOMENTUM)
                var one_m = Scalar[DT](1.0) - mom
                for c in range(Self.C_):
                    var g = g_p[unsafe_offset=c]
                    var bt = b_p[unsafe_offset=c]
                    var mean: Scalar[DT] = 0.0
                    for b in range(B):
                        var bb = b * Self.FLAT_DIM
                        for s in range(Self.SPATIAL):
                            mean += in_p[
                                unsafe_offset=bb
                                + _bn_off[Self.LAYOUT, Self.C_, Self.SPATIAL](
                                    c, s
                                )
                            ]
                    mean *= inv_n
                    var var_: Scalar[DT] = 0.0
                    for b in range(B):
                        var bb = b * Self.FLAT_DIM
                        for s in range(Self.SPATIAL):
                            var d = (
                                in_p[
                                    unsafe_offset=bb
                                    + _bn_off[
                                        Self.LAYOUT, Self.C_, Self.SPATIAL
                                    ](c, s)
                                ]
                                - mean
                            )
                            var_ += d * d
                    var_ *= inv_n
                    var inv_std = Scalar[DT](1.0) / sqrt(var_ + eps)
                    inv_v[c] = inv_std
                    for b in range(B):
                        var bb = b * Self.FLAT_DIM
                        for s in range(Self.SPATIAL):
                            var off = bb + _bn_off[
                                Self.LAYOUT, Self.C_, Self.SPATIAL
                            ](c, s)
                            var xh = (in_p[unsafe_offset=off] - mean) * inv_std
                            xhat_p[unsafe_offset=off] = xh
                            out_p[unsafe_offset=off] = g * xh + bt
                    rm_v[c] = one_m * rm_v[c] + mom * mean
                    rv_v[c] = one_m * rv_v[c] + mom * var_
                self.cache_is_training = True
            else:
                # Eval: normalize with running stats (constants). Cache xhat +
                # inv_std so an eval-mode backward can run without batch
                # reductions (gi = g·inv_std·dy) — used by the frozen-backbone
                # perceptual loss, which backprops through BN in eval mode.
                self.cache_xhat.ensure(B * Self.FLAT_DIM)
                self.cache_inv_std.ensure(Self.C_)
                var xhat_e = self.cache_xhat.data.unsafe_ptr()
                var inv_e = TileTensor(
                    self.cache_inv_std.data, row_major[Self.C_]()
                )
                for c in range(Self.C_):
                    var rm = rm_v[c]
                    var rv = rv_v[c]
                    var inv_std = Scalar[DT](1.0) / sqrt(rv + eps)
                    inv_e[c] = inv_std
                    var g = g_p[unsafe_offset=c]
                    var bt = b_p[unsafe_offset=c]
                    for b in range(B):
                        var bb = b * Self.FLAT_DIM
                        # ⚠ VECTORISED FOR NCHW ONLY, and the layout is the
                        # whole reason: `_bn_off` is `c*SPATIAL + s` there, so
                        # the `s` axis is CONTIGUOUS and every coefficient in
                        # this loop (`rm`, `inv_std`, `g`, `bt`) is a per-
                        # CHANNEL constant — a broadcast against a contiguous
                        # run, which is the shape SIMD wants. NHWC is
                        # `s*C + c`: stride-C loads, where a vector load would
                        # gather across channels and be slower than the scalar
                        # loop it replaced.
                        comptime if Self.LAYOUT == LAYOUT_NCHW:
                            var base = bb + c * Self.SPATIAL
                            var rmv = SIMD[DT, CPU_SIMD_W](rm)
                            var isv = SIMD[DT, CPU_SIMD_W](inv_std)
                            var gv = SIMD[DT, CPU_SIMD_W](g)
                            var btv = SIMD[DT, CPU_SIMD_W](bt)
                            var k = 0
                            while k + CPU_SIMD_W <= Self.SPATIAL:
                                var xv = in_p.unsafe_load[width=CPU_SIMD_W](
                                    base + k
                                )
                                var xhv = (xv - rmv) * isv
                                xhat_e.unsafe_store(base + k, xhv)
                                out_p.unsafe_store(base + k, gv * xhv + btv)
                                k += CPU_SIMD_W
                            while k < Self.SPATIAL:
                                var off = base + k
                                var xh = (
                                    in_p[unsafe_offset=off] - rm
                                ) * inv_std
                                xhat_e[unsafe_offset=off] = xh
                                out_p[unsafe_offset=off] = g * xh + bt
                                k += 1
                        else:
                            for s in range(Self.SPATIAL):
                                var off = bb + _bn_off[
                                    Self.LAYOUT, Self.C_, Self.SPATIAL
                                ](c, s)
                                var xh = (
                                    in_p[unsafe_offset=off] - rm
                                ) * inv_std
                                xhat_e[unsafe_offset=off] = xh
                                out_p[unsafe_offset=off] = g * xh + bt
                # ⚠ `xhat` IS STILL WRITTEN IN EVAL, and that is not an
                # oversight. Skipping it would halve this loop's memory traffic
                # — inference never reads it — but eval-mode BACKWARD is a
                # supported, GATED path here: the frozen-backbone perceptual
                # loss backprops through BN in eval mode, and
                # `test_batch_norm_2d_eval_vjp.mojo` pins it. Dropping the
                # write would not fail that gate loudly; it would produce a
                # wrong `dgamma`. It needs a real inference-mode flag, not a
                # guess about who is calling.
                self.cache_is_training = False
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.FLAT_DIM)
            comptime l2d = RowMajor[B, Self.FLAT_DIM]
            comptime lc = RowMajor[Self.C_]
            if self.training and not self.frozen:
                self.cache_xhat.ensure_gpu(c, B * Self.FLAT_DIM)
                comptime if Self.LAYOUT == LAYOUT_NHWC:
                    # Coalesced transposed reduction (channels-last perf path).
                    comptime CH = BN2D_NHWC_CHUNKS
                    comptime R = B * Self.SPATIAL
                    comptime RC = B * Self.FLAT_DIM
                    comptime lrc = RowMajor[RC]
                    comptime if Self.USE_2D_NHWC:
                        # NVIDIA occupancy path: per-row-group partials + parallel
                        # block.sum finalize (grid=C). See BN2D_NHWC_BLK note.
                        comptime ROWS = BN2D_NHWC_BLK // Self.C_
                        comptime P = CH * ROWS
                        comptime lpr2 = RowMajor[CH * BN2D_NHWC_BLK]
                        c.enqueue_function[
                            _bn2d_nhwc_partial_stats_2d[Self.C_, R, CH]
                        ](
                            in0.tt["gpu", RC](),
                            self.bn_psum.tt["gpu", CH * BN2D_NHWC_BLK](),
                            self.bn_psumsq.tt["gpu", CH * BN2D_NHWC_BLK](),
                            grid_dim=CH,
                            block_dim=BN2D_NHWC_BLK,
                        )
                        c.enqueue_function[
                            _bn2d_nhwc_finalize_stats_2d[
                                Self.C_, R, P, Self.EPSILON
                            ]
                        ](
                            self.bn_psum.tt["gpu", CH * BN2D_NHWC_BLK](),
                            self.bn_psumsq.tt["gpu", CH * BN2D_NHWC_BLK](),
                            self.cache_mean.tt["gpu", Self.C_](),
                            self.cache_var.tt["gpu", Self.C_](),
                            self.cache_inv_std.tt["gpu", Self.C_](),
                            grid_dim=Self.C_,
                            block_dim=BN2D_TPB,
                        )
                    else:
                        comptime lprn = RowMajor[CH * Self.C_]
                        c.enqueue_function[
                            _bn2d_nhwc_partial_stats[Self.C_, R, CH]
                        ](
                            in0.tt["gpu", RC](),
                            self.bn_psum.tt["gpu", CH * Self.C_](),
                            self.bn_psumsq.tt["gpu", CH * Self.C_](),
                            grid_dim=CH,
                            block_dim=Self.C_,
                        )
                        c.enqueue_function[
                            _bn2d_nhwc_finalize_stats[
                                Self.C_, R, CH, Self.EPSILON
                            ]
                        ](
                            self.bn_psum.tt["gpu", CH * Self.C_](),
                            self.bn_psumsq.tt["gpu", CH * Self.C_](),
                            self.cache_mean.tt["gpu", Self.C_](),
                            self.cache_var.tt["gpu", Self.C_](),
                            self.cache_inv_std.tt["gpu", Self.C_](),
                            grid_dim=(Self.C_ + BN2D_TPB - 1) // BN2D_TPB,
                            block_dim=BN2D_TPB,
                        )
                    c.enqueue_function[
                        _bn2d_update_running_kernel[Self.C_, Self.MOMENTUM]
                    ](
                        self.cache_mean.tt["gpu", Self.C_](),
                        self.cache_var.tt["gpu", Self.C_](),
                        self.running_mean.t.tt["gpu", Self.C_](),
                        self.running_var.t.tt["gpu", Self.C_](),
                        grid_dim=Self.C_,
                        block_dim=1,
                    )
                    c.enqueue_function[_bn2d_nhwc_normalize[Self.C_, RC]](
                        in0.tt["gpu", RC](),
                        out.tt["gpu", RC](),
                        self.gamma.val.tt["gpu", Self.C_](),
                        self.beta.val.tt["gpu", Self.C_](),
                        self.cache_mean.tt["gpu", Self.C_](),
                        self.cache_inv_std.tt["gpu", Self.C_](),
                        self.cache_xhat.tt["gpu", RC](),
                        grid_dim=(RC + BN2D_TPB - 1) // BN2D_TPB,
                        block_dim=BN2D_TPB,
                    )
                else:
                    comptime G = B if B < BN2D_RBLOCKS else BN2D_RBLOCKS
                    comptime lpr = RowMajor[Self.C_ * G]
                    # Pass 1: partial Σx, Σx².
                    c.enqueue_function[
                        _bn2d_partial_stats_kernel[
                            B,
                            Self.C_,
                            Self.SPATIAL,
                            Self.FLAT_DIM,
                            G,
                            Self.LAYOUT,
                        ]
                    ](
                        in0.tt["gpu", B, Self.FLAT_DIM](),
                        self.bn_psum.tt["gpu", Self.C_ * G](),
                        self.bn_psumsq.tt["gpu", Self.C_ * G](),
                        grid_dim=Self.C_ * G,
                        block_dim=BN2D_TPB,
                    )
                    # Pass 2: mean/var/inv_std (caches only — NO running write).
                    c.enqueue_function[
                        _bn2d_finalize_stats_kernel[
                            B,
                            Self.C_,
                            Self.SPATIAL,
                            G,
                            Self.EPSILON,
                        ]
                    ](
                        self.bn_psum.tt["gpu", Self.C_ * G](),
                        self.bn_psumsq.tt["gpu", Self.C_ * G](),
                        self.cache_mean.tt["gpu", Self.C_](),
                        self.cache_var.tt["gpu", Self.C_](),
                        self.cache_inv_std.tt["gpu", Self.C_](),
                        grid_dim=Self.C_,
                        block_dim=1,
                    )
                    # Pass 2b: running-stat EMA in a DEDICATED kernel (see kernel
                    # docstring — folding it into pass 2 dropped the stores on
                    # NVIDIA at B>64, pinning running stats at init).
                    c.enqueue_function[
                        _bn2d_update_running_kernel[Self.C_, Self.MOMENTUM]
                    ](
                        self.cache_mean.tt["gpu", Self.C_](),
                        self.cache_var.tt["gpu", Self.C_](),
                        self.running_mean.t.tt["gpu", Self.C_](),
                        self.running_var.t.tt["gpu", Self.C_](),
                        grid_dim=Self.C_,
                        block_dim=1,
                    )
                    # Pass 3: normalize + cache x̂.
                    c.enqueue_function[
                        _bn2d_normalize_kernel[
                            B,
                            Self.C_,
                            Self.SPATIAL,
                            Self.FLAT_DIM,
                            G,
                            Self.LAYOUT,
                        ]
                    ](
                        in0.tt["gpu", B, Self.FLAT_DIM](),
                        out.tt["gpu", B, Self.FLAT_DIM](),
                        self.gamma.val.tt["gpu", Self.C_](),
                        self.beta.val.tt["gpu", Self.C_](),
                        self.cache_mean.tt["gpu", Self.C_](),
                        self.cache_inv_std.tt["gpu", Self.C_](),
                        self.cache_xhat.tt["gpu", B, Self.FLAT_DIM](),
                        grid_dim=Self.C_ * G,
                        block_dim=BN2D_TPB,
                    )
                self.cache_is_training = True
            else:
                c.enqueue_function[
                    _bn2d_forward_eval_kernel[
                        B,
                        Self.C_,
                        Self.SPATIAL,
                        Self.FLAT_DIM,
                        Self.EPSILON,
                        Self.LAYOUT,
                    ]
                ](
                    in0.tt["gpu", B, Self.FLAT_DIM](),
                    out.tt["gpu", B, Self.FLAT_DIM](),
                    self.gamma.val.tt["gpu", Self.C_](),
                    self.beta.val.tt["gpu", Self.C_](),
                    self.running_mean.t.tt["gpu", Self.C_](),
                    self.running_var.t.tt["gpu", Self.C_](),
                    grid_dim=Self.C_,
                    block_dim=BN2D_TPB,
                )
                self.cache_is_training = False

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Backward, then the frozen-mode affine-gradient reset.

        ⚠ A WRAPPER, not a tail added to the body, because `_vjp_impl` RETURNS
        EARLY from both eval-mode branches (CPU and GPU) — which is exactly the
        path frozen mode takes, so a tail would be unreachable in the only mode
        that needs it. Written as a tail first; the gate caught it.

        Zeroing after the fact rather than skipping the five write sites (one
        CPU, four GPU kernels) is deliberate: a skip that misses one is silent,
        the parameter keeps training, and the run reads as "freezing did not
        help". Zeroing is also correct under accumulation, which matters —
        `Tokenwise[N_CAM, backbone]` runs this module once per camera and the
        gradients add across those calls.
        """
        self._vjp_impl[target, B, ofi, ogi, POLICY](
            forward_input, grad_output, grad_inputs, ctx
        )
        if not self.frozen:
            return
        # A released gradient (`Param.freeze`) has nothing to zero, and
        # `ensure_gpu` below would allocate it again.
        if self.gamma.grd.n == 0 or self.beta.grd.n == 0:
            return
        # gamma/beta are `register_buffer` in the reference — constants.
        comptime if target == "cpu":
            for k in range(Self.C_):
                self.gamma.grd.data[k] = Scalar[DT](0.0)
                self.beta.grd.data[k] = Scalar[DT](0.0)
        else:
            var cz = ctx.value()
            self.gamma.grd.ensure_gpu(cz, Self.C_)
            self.beta.grd.ensure_gpu(cz, Self.C_)
            self.gamma.grd.dev.value().enqueue_fill(Scalar[DT](0.0))
            self.beta.grd.dev.value().enqueue_fill(Scalar[DT](0.0))

    def _vjp_impl[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # Eval-mode (running-stat) backward is implemented on both CPU and GPU
        # (gi = γ·inv_std_running·dy); no training-mode forward required.
        # AMP §3 fp32-internal (forward_input unused, as in the fp32 body).
        comptime if Self.ACT_DT == DT:
            ref god = rebind[Tensor](grad_output)
            ref gind = rebind[Tensor](grad_inputs[0])
            ref find = rebind[Tensor](forward_input[0])
            self._vjp_f32[target, B](god, gind, find, ctx)
        else:
            comptime N = B * Self.FLAT_DIM
            # LOCAL fp32 scratch (not self-fields → no mut-self aliasing).
            var go_f32 = Tensor()
            go_f32.ensure[target](N, ctx)
            var gin_f32 = Tensor()
            gin_f32.ensure[target](N, ctx)
            ref gin = grad_inputs[0]
            gin.ensure[target](N, ctx)
            comptime if target == "cpu":
                for i in range(N):
                    go_f32.data[i] = grad_output.data[i].cast[DT]()
            else:
                var c = ctx.value()
                c.enqueue_function[_cast_b2f_kernel](
                    grad_output.dev.value(),
                    go_f32.dev.value(),
                    Int64(N),
                    grid_dim=(N + 255) // 256,
                    block_dim=256,
                )
            ref find2 = rebind[Tensor](forward_input[0])
            self._vjp_f32[target, B](go_f32, gin_f32, find2, ctx)
            comptime if target == "cpu":
                for i in range(N):
                    gin.data[i] = gin_f32.data[i].cast[Self.ACT_DT]()
            else:
                var c = ctx.value()
                c.enqueue_function[_cast_f2b_kernel](
                    gin_f32.dev.value(),
                    gin.dev.value(),
                    Int64(N),
                    grid_dim=(N + 255) // 256,
                    block_dim=256,
                )

    def _vjp_f32[target: StaticString, B: Int](
        mut self,
        mut grad_output: Tensor,
        mut gin: Tensor,
        mut fin: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`fin` is the forward INPUT. Only the GPU eval-mode branch reads it,
        to recompute `xhat` for the affine gradients — the eval forward does not
        cache it on GPU, deliberately (see `_bn2d_eval_bwd_affine_kernel`)."""
        comptime if target == "cpu":
            gin.ensure(B * Self.FLAT_DIM)
            var go_p = grad_output.data.unsafe_ptr()
            var gi_p = gin.data.unsafe_ptr()
            var g_p = self.gamma.val.data.unsafe_ptr()
            var dg_p = self.gamma.grd.data.unsafe_ptr()
            var db_p = self.beta.grd.data.unsafe_ptr()
            var xhat_p = self.cache_xhat.data.unsafe_ptr()
            var inv_v = TileTensor(
                self.cache_inv_std.data, row_major[Self.C_]()
            )
            if not self.cache_is_training:
                # Eval-mode backward: running mean/var are constants, so there is
                # no batch coupling — gi = g·inv_std·dy (cf. the batch-stat path
                # below which subtracts the m1/m2 batch reductions). γ/β grads are
                # still accumulated (harmless for a frozen backbone).
                for c in range(Self.C_):
                    var g_e = g_p[unsafe_offset=c]
                    var inv_e = inv_v[c]
                    var dg_e: Scalar[DT] = 0.0
                    var db_e: Scalar[DT] = 0.0
                    for b in range(B):
                        var bb = b * Self.FLAT_DIM
                        for s in range(Self.SPATIAL):
                            var off = bb + _bn_off[
                                Self.LAYOUT, Self.C_, Self.SPATIAL
                            ](c, s)
                            var dy = go_p[unsafe_offset=off]
                            gi_p[unsafe_offset=off] = inv_e * dy * g_e
                            dg_e += dy * xhat_p[unsafe_offset=off]
                            db_e += dy
                    dg_p[unsafe_offset=c] += dg_e
                    db_p[unsafe_offset=c] += db_e
                return
            var inv_n = Scalar[DT](1.0) / Scalar[DT](Float64(B * Self.SPATIAL))
            for c in range(Self.C_):
                var g = g_p[unsafe_offset=c]
                var inv_std = inv_v[c]
                var sum_dxhat: Scalar[DT] = 0.0
                var sum_dxhat_xhat: Scalar[DT] = 0.0
                var d_gamma: Scalar[DT] = 0.0
                var d_beta: Scalar[DT] = 0.0
                for b in range(B):
                    var bb = b * Self.FLAT_DIM
                    for s in range(Self.SPATIAL):
                        var off = bb + _bn_off[
                            Self.LAYOUT, Self.C_, Self.SPATIAL
                        ](c, s)
                        var dy = go_p[unsafe_offset=off]
                        var xh = xhat_p[unsafe_offset=off]
                        var dxhat = dy * g
                        sum_dxhat += dxhat
                        sum_dxhat_xhat += dxhat * xh
                        d_gamma += dy * xh
                        d_beta += dy
                var m1 = sum_dxhat * inv_n
                var m2 = sum_dxhat_xhat * inv_n
                for b in range(B):
                    var bb = b * Self.FLAT_DIM
                    for s in range(Self.SPATIAL):
                        var off = bb + _bn_off[
                            Self.LAYOUT, Self.C_, Self.SPATIAL
                        ](c, s)
                        var dy = go_p[unsafe_offset=off]
                        var xh = xhat_p[unsafe_offset=off]
                        var dxhat = dy * g
                        gi_p[unsafe_offset=off] = inv_std * (dxhat - m1 - xh * m2)
                dg_p[unsafe_offset=c] += d_gamma
                db_p[unsafe_offset=c] += d_beta
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.FLAT_DIM)
            comptime l2d = RowMajor[B, Self.FLAT_DIM]
            comptime lc = RowMajor[Self.C_]
            if not self.cache_is_training:
                # Eval-mode backward. Running stats are constants, so the input
                # gradient has no batch coupling — but γ and β still receive
                # gradient, and this path used to drop them silently while the
                # CPU accumulated them. See
                # `tests/nn/test_batch_norm_2d_eval_backward_gpu.mojo`.
                c.enqueue_function[
                    _bn2d_eval_bwd_affine_kernel[
                        B, Self.C_, Self.SPATIAL, Self.FLAT_DIM,
                        Self.EPSILON, Self.LAYOUT,
                    ]
                ](
                    grad_output.tt["gpu", B, Self.FLAT_DIM](),
                    fin.tt["gpu", B, Self.FLAT_DIM](),
                    self.gamma.val.tt["gpu", Self.C_](),
                    self.running_mean.t.tt["gpu", Self.C_](),
                    self.running_var.t.tt["gpu", Self.C_](),
                    gin.tt["gpu", B, Self.FLAT_DIM](),
                    self.gamma.grd.tt["gpu", Self.C_](),
                    self.beta.grd.tt["gpu", Self.C_](),
                    grid_dim=Self.C_,
                    block_dim=BN2D_TPB,
                )
                return
            comptime if Self.LAYOUT == LAYOUT_NHWC:
                # Coalesced transposed backward (channels-last perf path).
                comptime CH = BN2D_NHWC_CHUNKS
                comptime R = B * Self.SPATIAL
                comptime RC = B * Self.FLAT_DIM
                comptime lrc = RowMajor[RC]
                comptime if Self.USE_2D_NHWC:
                    comptime ROWS = BN2D_NHWC_BLK // Self.C_
                    comptime P = CH * ROWS
                    comptime lpr2 = RowMajor[CH * BN2D_NHWC_BLK]
                    c.enqueue_function[
                        _bn2d_nhwc_bwd_partial_2d[Self.C_, R, CH]
                    ](
                        grad_output.tt["gpu", RC](),
                        self.gamma.val.tt["gpu", Self.C_](),
                        self.cache_xhat.tt["gpu", RC](),
                        self.bn_psum.tt["gpu", CH * BN2D_NHWC_BLK](),
                        self.bn_psumsq.tt["gpu", CH * BN2D_NHWC_BLK](),
                        self.bn_pdg.tt["gpu", CH * BN2D_NHWC_BLK](),
                        self.bn_pdb.tt["gpu", CH * BN2D_NHWC_BLK](),
                        grid_dim=CH,
                        block_dim=BN2D_NHWC_BLK,
                    )
                    c.enqueue_function[
                        _bn2d_nhwc_bwd_finalize_2d[Self.C_, R, P, "all"]
                    ](
                        self.bn_psum.tt["gpu", CH * BN2D_NHWC_BLK](),
                        self.bn_psumsq.tt["gpu", CH * BN2D_NHWC_BLK](),
                        self.bn_pdg.tt["gpu", CH * BN2D_NHWC_BLK](),
                        self.bn_pdb.tt["gpu", CH * BN2D_NHWC_BLK](),
                        self.bn_m1.tt["gpu", Self.C_](),
                        self.bn_m2.tt["gpu", Self.C_](),
                        self.gamma.grd.tt["gpu", Self.C_](),
                        self.beta.grd.tt["gpu", Self.C_](),
                        grid_dim=Self.C_,
                        block_dim=BN2D_TPB,
                    )
                else:
                    comptime lprn = RowMajor[CH * Self.C_]
                    c.enqueue_function[
                        _bn2d_nhwc_bwd_partial[Self.C_, R, CH]
                    ](
                        grad_output.tt["gpu", RC](),
                        self.gamma.val.tt["gpu", Self.C_](),
                        self.cache_xhat.tt["gpu", RC](),
                        self.bn_psum.tt["gpu", CH * Self.C_](),
                        self.bn_psumsq.tt["gpu", CH * Self.C_](),
                        self.bn_pdg.tt["gpu", CH * Self.C_](),
                        self.bn_pdb.tt["gpu", CH * Self.C_](),
                        grid_dim=CH,
                        block_dim=Self.C_,
                    )
                    c.enqueue_function[
                        _bn2d_nhwc_bwd_finalize[Self.C_, R, CH, "all"]
                    ](
                        self.bn_psum.tt["gpu", CH * Self.C_](),
                        self.bn_psumsq.tt["gpu", CH * Self.C_](),
                        self.bn_pdg.tt["gpu", CH * Self.C_](),
                        self.bn_pdb.tt["gpu", CH * Self.C_](),
                        self.bn_m1.tt["gpu", Self.C_](),
                        self.bn_m2.tt["gpu", Self.C_](),
                        self.gamma.grd.tt["gpu", Self.C_](),
                        self.beta.grd.tt["gpu", Self.C_](),
                        grid_dim=(Self.C_ + BN2D_TPB - 1) // BN2D_TPB,
                        block_dim=BN2D_TPB,
                    )
                c.enqueue_function[_bn2d_nhwc_bwd_scatter[Self.C_, RC]](
                    grad_output.tt["gpu", RC](),
                    self.gamma.val.tt["gpu", Self.C_](),
                    self.cache_xhat.tt["gpu", RC](),
                    self.cache_inv_std.tt["gpu", Self.C_](),
                    self.bn_m1.tt["gpu", Self.C_](),
                    self.bn_m2.tt["gpu", Self.C_](),
                    gin.tt["gpu", RC](),
                    grid_dim=(RC + BN2D_TPB - 1) // BN2D_TPB,
                    block_dim=BN2D_TPB,
                )
            else:
                comptime G = B if B < BN2D_RBLOCKS else BN2D_RBLOCKS
                comptime lpr = RowMajor[Self.C_ * G]
                # Pass 1: partial Σdx̂, Σdx̂·x̂, Σdγ, Σdβ (reuse psum/psumsq).
                c.enqueue_function[
                    _bn2d_bwd_partial_kernel[
                        B,
                        Self.C_,
                        Self.SPATIAL,
                        Self.FLAT_DIM,
                        G,
                        Self.LAYOUT,
                    ]
                ](
                    grad_output.tt["gpu", B, Self.FLAT_DIM](),
                    self.gamma.val.tt["gpu", Self.C_](),
                    self.cache_xhat.tt["gpu", B, Self.FLAT_DIM](),
                    self.bn_psum.tt["gpu", Self.C_ * G](),
                    self.bn_psumsq.tt["gpu", Self.C_ * G](),
                    self.bn_pdg.tt["gpu", Self.C_ * G](),
                    self.bn_pdb.tt["gpu", Self.C_ * G](),
                    grid_dim=Self.C_ * G,
                    block_dim=BN2D_TPB,
                )
                # Pass 2: m1, m2 + accumulate grad_gamma/beta (mode=all).
                c.enqueue_function[
                    _bn2d_bwd_finalize_kernel[
                        B,
                        Self.C_,
                        Self.SPATIAL,
                        G,
                        "all",
                    ]
                ](
                    self.bn_psum.tt["gpu", Self.C_ * G](),
                    self.bn_psumsq.tt["gpu", Self.C_ * G](),
                    self.bn_pdg.tt["gpu", Self.C_ * G](),
                    self.bn_pdb.tt["gpu", Self.C_ * G](),
                    self.bn_m1.tt["gpu", Self.C_](),
                    self.bn_m2.tt["gpu", Self.C_](),
                    self.gamma.grd.tt["gpu", Self.C_](),
                    self.beta.grd.tt["gpu", Self.C_](),
                    grid_dim=Self.C_,
                    block_dim=1,
                )
                # Pass 3: grad_input scatter.
                c.enqueue_function[
                    _bn2d_bwd_scatter_kernel[
                        B,
                        Self.C_,
                        Self.SPATIAL,
                        Self.FLAT_DIM,
                        G,
                        Self.LAYOUT,
                    ]
                ](
                    grad_output.tt["gpu", B, Self.FLAT_DIM](),
                    self.gamma.val.tt["gpu", Self.C_](),
                    self.cache_xhat.tt["gpu", B, Self.FLAT_DIM](),
                    self.cache_inv_std.tt["gpu", Self.C_](),
                    self.bn_m1.tt["gpu", Self.C_](),
                    self.bn_m2.tt["gpu", Self.C_](),
                    gin.tt["gpu", B, Self.FLAT_DIM](),
                    grid_dim=Self.C_ * G,
                    block_dim=BN2D_TPB,
                )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the Param fields).
