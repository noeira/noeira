"""Parameter-gradient reductions: the old kernels against the shared
two-pass helpers in `nn/primitives/linear.mojo`, on the same buffers, at the
shapes their users run.

Each old kernel is a LOCAL COPY of what the primitive launched before the fix
(`noeira-docs/NN_PARAM_GRAD_REDUCTIONS_FIX.md`), so the comparison survives
the fix. Old shapes:
  (a′) one block per column, threads striding over the rows, `block.sum`;
  (a)  one thread per column walking every row.

Timing: N launches enqueued between two syncs, divided by N (a single synced
launch on Metal measures the ~250 us round trip). Old and new interleave, 3
rounds, best of each. Every case first checks new against old (both from
zero, one call), in units of the largest |old| value.

Decision rule: fix if new is >= 1.3x faster AND saves >= 20 us per call.

    pixi run -e apple mojo run -I . benchmarks/bench_param_grad_reductions.mojo
"""

from std.math import abs
from std.random import seed, random_float64
from std.time import perf_counter_ns
from max.gpu import thread_idx, block_idx, global_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext

from noeira.core.tiles import Tile
from noeira.nn.constants import DT, LAYOUT_NCHW, LAYOUT_NHWC
from noeira.nn.core.element_op import ElementOp
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.fill import fill_dev
from noeira.nn.primitives.ops.mish_op import MishOp
from noeira.nn.primitives.conv2d import _out_off
from noeira.nn.primitives.layer_norm_act import _act_cache_of
from noeira.nn.primitives.linear import (
    enqueue_bias_grad,
    enqueue_dot_grad,
    enqueue_dot_bias_grad,
    enqueue_channel_grad,
    enqueue_channel_dot_grad,
)

comptime BF16 = DType.bfloat16
comptime OLD_TPB = 128
comptime N_CALLS = 50
comptime ROUNDS = 3


# ── old kernels (local copies) ──────────────────────────────────────────────
def _old_conv_db[
    BATCH: Int, OC: Int, SO: Int, OUT_FLAT: Int, ADT: DType, LAYOUT: Int
](
    grad_output: Tile[ADT, BATCH, OUT_FLAT],
    grad_bias: Tile[DT, OC],
):
    """conv2d `_backward_db_kernel`: grid = OC blocks over B·SO values."""
    var oc = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if oc >= OC:
        return
    var n_eff = BATCH * SO
    var my_acc: Scalar[DT] = 0
    var idx = t
    while idx < n_eff:
        var b = idx // SO
        var s_pos = idx % SO
        my_acc += rebind[Scalar[ADT]](
            grad_output[b, _out_off[LAYOUT, OC, SO](oc, s_pos)]
        ).cast[DT]()
        idx += OLD_TPB
    var total = block.sum[block_size=OLD_TPB, broadcast=False](val=my_acc)
    if t == 0:
        grad_bias[oc] = rebind[Scalar[DT]](grad_bias[oc]) + total[0]


def _old_col_sum[
    BATCH: Int, DIM: Int, ADT: DType
](
    grad_output: Tile[ADT, BATCH, DIM],
    grad_bias: Tile[DT, DIM],
):
    """bias_add `_bias_add_dbias_kernel` (and lstm `_lstm_db_kernel`,
    noisy `_grad_b_pair_reduce_kernel`'s sum): grid = DIM blocks."""
    var col = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if col >= DIM:
        return
    var my_db: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        my_db += rebind[Scalar[ADT]](grad_output[bi, col]).cast[DT]()
        bi += OLD_TPB
    var total_db = block.sum[block_size=OLD_TPB, broadcast=False](val=my_db)
    if t == 0:
        grad_bias[col] = rebind[Scalar[DT]](grad_bias[col]) + total_db[0]


def _old_dot_col_sum[
    BATCH: Int, DIM: Int
](
    grad_output: Tile[DT, BATCH, DIM],
    cache_norm: Tile[DT, BATCH, DIM],
    grad_gamma: Tile[DT, DIM],
):
    """rms_norm `_rms_norm_backward_dgamma_kernel`: grid = DIM blocks."""
    var col = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if col >= DIM:
        return
    var my_dg: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        my_dg += rebind[Scalar[DT]](grad_output[bi, col]) * rebind[Scalar[DT]](
            cache_norm[bi, col]
        )
        bi += OLD_TPB
    var total_dg = block.sum[block_size=OLD_TPB, broadcast=False](val=my_dg)
    if t == 0:
        grad_gamma[col] = rebind[Scalar[DT]](grad_gamma[col]) + total_dg[0]


def _old_conv_rms_dgamma[
    B: Int, C: Int, HW: Int
](
    grad_output: Tile[DT, B, C * HW],
    cache_norm: Tile[DT, B, C * HW],
    grad_gamma: Tile[DT, C],
):
    """conv_rms_norm `_conv_rms_backward_dgamma_kernel`: grid = C blocks."""
    var c = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if c >= C:
        return
    var my_dg: Scalar[DT] = 0.0
    var i = t
    while i < B * HW:
        var b = i // HW
        var p = i % HW
        my_dg += rebind[Scalar[DT]](grad_output[b, c * HW + p]) * rebind[
            Scalar[DT]
        ](cache_norm[b, c * HW + p])
        i += OLD_TPB
    var total_dg = block.sum[block_size=OLD_TPB, broadcast=False](val=my_dg)
    if t == 0:
        grad_gamma[c] = rebind[Scalar[DT]](grad_gamma[c]) + total_dg[0]


def _old_thread_col_sum[
    BATCH: Int, OUT: Int
](
    go: Tile[DT, BATCH * OUT],
    grad_b: Tile[DT, OUT],
):
    """block_linear `_bl_dbias_kernel`: one thread per column."""
    var j = Int(global_idx.x)
    if j >= OUT:
        return
    var acc: Scalar[DT] = 0.0
    for b in range(BATCH):
        acc += rebind[Scalar[DT]](go[b * OUT + j])
    grad_b[j] = rebind[Scalar[DT]](grad_b[j]) + acc


def _old_ln_act_dparams[
    BATCH: Int, DIM: Int, OP: ElementOp
](
    grad_output: Tile[DT, BATCH, DIM],
    gamma: Tile[DT, DIM],
    beta: Tile[DT, DIM],
    cache_xhat: Tile[DT, BATCH, DIM],
    grad_gamma: Tile[DT, DIM],
    grad_beta: Tile[DT, DIM],
):
    """layer_norm_act `_ln_act_backward_dparams_kernel`: grid = DIM blocks."""
    var col = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if col >= DIM:
        return
    var gm = rebind[Scalar[DT]](gamma[col])
    var bt = rebind[Scalar[DT]](beta[col])
    var my_dg: Scalar[DT] = 0.0
    var my_db: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        var go_a = rebind[Scalar[DT]](grad_output[bi, col])
        var xh = rebind[Scalar[DT]](cache_xhat[bi, col])
        var go = OP.backward_scalar(_act_cache_of[OP](gm * xh + bt), go_a)
        my_dg += go * xh
        my_db += go
        bi += OLD_TPB
    var total_dg = block.sum[block_size=OLD_TPB, broadcast=False](val=my_dg)
    var total_db = block.sum[block_size=OLD_TPB, broadcast=False](val=my_db)
    if t == 0:
        grad_gamma[col] = rebind[Scalar[DT]](grad_gamma[col]) + total_dg[0]
        grad_beta[col] = rebind[Scalar[DT]](grad_beta[col]) + total_db[0]


def _old_bn1d_bwd[
    BATCH: Int, DIM: Int
](
    grad_output: Tile[DT, BATCH, DIM],
    gamma: Tile[DT, DIM],
    cache_xhat: Tile[DT, BATCH, DIM],
    cache_inv_std: Tile[DT, DIM],
    grad_input: Tile[DT, BATCH, DIM],
    grad_gamma: Tile[DT, DIM],
    grad_beta: Tile[DT, DIM],
):
    """batch_norm_1d `_bn1d_backward_kernel`: grid = DIM blocks, the column
    sums and dx in one launch (two uncoalesced passes over the rows)."""
    var f = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if f >= DIM:
        return
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(BATCH))
    var g = rebind[Scalar[DT]](gamma[f])
    var inv_std = rebind[Scalar[DT]](cache_inv_std[f])
    var my_sum_dxhat: Scalar[DT] = 0.0
    var my_sum_dxhat_xhat: Scalar[DT] = 0.0
    var my_dgamma: Scalar[DT] = 0.0
    var my_dbeta: Scalar[DT] = 0.0
    var b = t
    while b < BATCH:
        var dy = rebind[Scalar[DT]](grad_output[b, f])
        var xh = rebind[Scalar[DT]](cache_xhat[b, f])
        var dxhat = dy * g
        my_sum_dxhat += dxhat
        my_sum_dxhat_xhat += dxhat * xh
        my_dgamma += dy * xh
        my_dbeta += dy
        b += OLD_TPB
    var sum_dxhat = block.sum[block_size=OLD_TPB, broadcast=True](val=my_sum_dxhat)
    var sum_dxhat_xhat = block.sum[block_size=OLD_TPB, broadcast=True](
        val=my_sum_dxhat_xhat
    )
    var d_gamma_tot = block.sum[block_size=OLD_TPB, broadcast=False](val=my_dgamma)
    var d_beta_tot = block.sum[block_size=OLD_TPB, broadcast=False](val=my_dbeta)
    if t == 0:
        grad_gamma[f] = rebind[Scalar[DT]](grad_gamma[f]) + d_gamma_tot[0]
        grad_beta[f] = rebind[Scalar[DT]](grad_beta[f]) + d_beta_tot[0]
    var m1 = sum_dxhat * inv_n
    var m2 = sum_dxhat_xhat * inv_n
    b = t
    while b < BATCH:
        var dy = rebind[Scalar[DT]](grad_output[b, f])
        var xh = rebind[Scalar[DT]](cache_xhat[b, f])
        grad_input[b, f] = inv_std * (dy * g - m1 - xh * m2)
        b += OLD_TPB


def _new_bn1d_dx[
    BATCH: Int, DIM: Int
](
    grad_output: Tile[DT, BATCH, DIM],
    gamma: Tile[DT, DIM],
    cache_xhat: Tile[DT, BATCH, DIM],
    cache_inv_std: Tile[DT, DIM],
    s_g: Tile[DT, DIM],
    s_b: Tile[DT, DIM],
    grad_input: Tile[DT, BATCH, DIM],
    grad_gamma: Tile[DT, DIM],
    grad_beta: Tile[DT, DIM],
):
    """Candidate BN1D dx: one thread per element (coalesced), from the column
    sums `s_g = Σ dy·x̂`, `s_b = Σ dy` the two-pass helper wrote; row 0 adds
    them to the γ / β grads."""
    var i = Int(global_idx.x)
    if i >= BATCH * DIM:
        return
    var b = i // DIM
    var f = i % DIM
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(BATCH))
    var g = rebind[Scalar[DT]](gamma[f])
    var sg = rebind[Scalar[DT]](s_g[f])
    var sb = rebind[Scalar[DT]](s_b[f])
    var dy = rebind[Scalar[DT]](grad_output[b, f])
    var xh = rebind[Scalar[DT]](cache_xhat[b, f])
    grad_input[b, f] = rebind[Scalar[DT]](cache_inv_std[f]) * (
        dy * g - g * sb * inv_n - xh * g * sg * inv_n
    )
    if b == 0:
        grad_gamma[f] = rebind[Scalar[DT]](grad_gamma[f]) + sg
        grad_beta[f] = rebind[Scalar[DT]](grad_beta[f]) + sb


# ── harness ─────────────────────────────────────────────────────────────────
struct Bufs(Movable):
    var go: Tensor
    var gob: TensorImpl[BF16]
    var x: Tensor
    var gamma: Tensor
    var beta: Tensor
    var g1: Tensor
    var g2: Tensor
    var rows: Tensor
    var part: Tensor
    var n: Int
    var m: Int

    def __init__(out self, c: DeviceContext, n: Int, m: Int) raises:
        """`n` input elements, `m` gradient elements."""
        self.n = n
        self.m = m
        self.go = Tensor.alloc(n)
        self.gob = TensorImpl[BF16].alloc(n)
        self.x = Tensor.alloc(n)
        for i in range(n):
            var v = Scalar[DT](random_float64(-1, 1))
            self.go.data[i] = v
            self.gob.data[i] = v.cast[BF16]()
            self.x.data[i] = Scalar[DT](random_float64(-2, 2))
        self.go.upload(c)
        self.gob.upload(c)
        self.x.upload(c)
        self.gamma = Tensor.alloc(m)
        self.beta = Tensor.alloc(m)
        for i in range(m):
            self.gamma.data[i] = Scalar[DT](random_float64(0.5, 1.5))
            self.beta.data[i] = Scalar[DT](random_float64(-0.5, 0.5))
        self.gamma.upload(c)
        self.beta.upload(c)
        self.g1 = Tensor.alloc_gpu(c, m)
        self.g2 = Tensor.alloc_gpu(c, m)
        self.rows = Tensor()
        self.part = Tensor()


comptime Launch = def(DeviceContext, mut Bufs) raises thin -> None


def _check[old: Launch, new: Launch](c: DeviceContext, mut b: Bufs) raises -> Float64:
    """max |new - old| / max |old| after one call each from zero (g1 only)."""
    b.g1 = Tensor.alloc_gpu(c, b.m)
    old(c, b)
    b.g1.download(c)
    var ref_ = b.g1.data.copy()
    b.g1 = Tensor.alloc_gpu(c, b.m)
    new(c, b)
    b.g1.download(c)
    var mx = 0.0
    var err = 0.0
    for i in range(b.m):
        mx = max(mx, abs(Float64(ref_[i])))
        err = max(err, abs(Float64(b.g1.data[i]) - Float64(ref_[i])))
    return err / (mx if mx > 0.0 else 1.0)


def _time_us[f: Launch](c: DeviceContext, mut b: Bufs) raises -> Float64:
    c.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(N_CALLS):
        f(c, b)
    c.synchronize()
    return Float64(perf_counter_ns() - t0) / 1e3 / Float64(N_CALLS)


def run[old: Launch, new: Launch](
    c: DeviceContext, name: String, shape: String, mut b: Bufs
) raises:
    var err = _check[old, new](c, b)
    _ = _time_us[old](c, b)  # warm-up
    _ = _time_us[new](c, b)
    var t_old = 1e30
    var t_new = 1e30
    for _ in range(ROUNDS):
        t_old = min(t_old, _time_us[old](c, b))
        t_new = min(t_new, _time_us[new](c, b))
    var ratio = t_old / t_new
    var saved = t_old - t_new
    var verdict = "FIX" if ratio >= 1.3 and saved >= 20.0 else "not worth it"
    print(
        "|", name, "|", shape, "|", round(t_old, 1), "|", round(t_new, 1),
        "|", round(ratio, 2), "x |", round(saved, 1), "|", verdict,
        "| err", err, "|",
    )


# ── cases ───────────────────────────────────────────────────────────────────
# #1 / #2 conv bias over B·SO (old grid = OC).
def old_conv_db[
    B: Int, OC: Int, SO: Int, ADT: DType, LAYOUT: Int
](c: DeviceContext, mut b: Bufs) raises:
    comptime if ADT == DT:
        c.enqueue_function[_old_conv_db[B, OC, SO, OC * SO, DT, LAYOUT]](
            b.go.tt["gpu", B, OC * SO](), b.g1.tt["gpu", OC](),
            grid_dim=OC, block_dim=OLD_TPB,
        )
    else:
        c.enqueue_function[_old_conv_db[B, OC, SO, OC * SO, BF16, LAYOUT]](
            b.gob.tt["gpu", B, OC * SO](), b.g1.tt["gpu", OC](),
            grid_dim=OC, block_dim=OLD_TPB,
        )


def new_conv_db[
    B: Int, OC: Int, SO: Int, ADT: DType, LAYOUT: Int
](c: DeviceContext, mut b: Bufs) raises:
    comptime if LAYOUT == LAYOUT_NHWC:
        comptime if ADT == DT:
            enqueue_bias_grad[DT](c, b.go.dev.value(), b.g1.dev.value(), B * SO, OC, b.part)
        else:
            enqueue_bias_grad[BF16](c, b.gob.dev.value(), b.g1.dev.value(), B * SO, OC, b.part)
    else:
        comptime if ADT == DT:
            enqueue_channel_grad[DT](
                c, b.go.dev.value(), b.g1.dev.value(), B, OC, SO, b.rows, b.part
            )
        else:
            enqueue_channel_grad[BF16](
                c, b.gob.dev.value(), b.g1.dev.value(), B, OC, SO, b.rows, b.part
            )


def case_conv_db[
    B: Int, OC: Int, SO: Int, ADT: DType, LAYOUT: Int
](c: DeviceContext, name: String) raises:
    var b = Bufs(c, B * OC * SO, OC)
    var lay = "NHWC" if LAYOUT == LAYOUT_NHWC else "NCHW"
    var dt = "bf16" if ADT == BF16 else "fp32"
    run[old_conv_db[B, OC, SO, ADT, LAYOUT], new_conv_db[B, OC, SO, ADT, LAYOUT]](
        c, name,
        String("B=", B, " OC=", OC, " SO=", SO, " ", lay, " ", dt), b,
    )


# #3 RMSNorm dγ = Σ go·n over rows.
def old_rms[B: Int, D: Int](c: DeviceContext, mut b: Bufs) raises:
    c.enqueue_function[_old_dot_col_sum[B, D]](
        b.go.tt["gpu", B, D](), b.x.tt["gpu", B, D](), b.g1.tt["gpu", D](),
        grid_dim=D, block_dim=OLD_TPB,
    )


def new_rms[B: Int, D: Int](c: DeviceContext, mut b: Bufs) raises:
    enqueue_dot_grad[DT](c, b.go.dev.value(), b.x.dev.value(), b.g1.dev.value(), B, D, b.part)


def case_rms[B: Int, D: Int](c: DeviceContext, name: String) raises:
    var b = Bufs(c, B * D, D)
    run[old_rms[B, D], new_rms[B, D]](c, name, String("R=", B, " C=", D), b)


# #4 ConvRMSNorm dγ over B·HW (NCHW).
def old_crms[B: Int, C: Int, HW: Int](c: DeviceContext, mut b: Bufs) raises:
    c.enqueue_function[_old_conv_rms_dgamma[B, C, HW]](
        b.go.tt["gpu", B, C * HW](), b.x.tt["gpu", B, C * HW](), b.g1.tt["gpu", C](),
        grid_dim=C, block_dim=OLD_TPB,
    )


def new_crms[B: Int, C: Int, HW: Int](c: DeviceContext, mut b: Bufs) raises:
    enqueue_channel_dot_grad(
        c, b.go.dev.value(), b.x.dev.value(), b.g1.dev.value(), B, C, HW, b.rows, b.part
    )


def case_crms[B: Int, C: Int, HW: Int](c: DeviceContext, name: String) raises:
    var b = Bufs(c, B * C * HW, C)
    run[old_crms[B, C, HW], new_crms[B, C, HW]](
        c, name, String("B=", B, " C=", C, " HW=", HW), b
    )


# #5 BlockLinear db (one thread per column).
def old_thread[B: Int, D: Int](c: DeviceContext, mut b: Bufs) raises:
    c.enqueue_function[_old_thread_col_sum[B, D]](
        b.go.tt["gpu", B * D](), b.g1.tt["gpu", D](),
        grid_dim=(D + 127) // 128, block_dim=128,
    )


# #6 / #9 plain column sum (old grid = DIM blocks).
def old_col[B: Int, D: Int, ADT: DType](c: DeviceContext, mut b: Bufs) raises:
    comptime if ADT == DT:
        c.enqueue_function[_old_col_sum[B, D, DT]](
            b.go.tt["gpu", B, D](), b.g1.tt["gpu", D](),
            grid_dim=D, block_dim=OLD_TPB,
        )
    else:
        c.enqueue_function[_old_col_sum[B, D, BF16]](
            b.gob.tt["gpu", B, D](), b.g1.tt["gpu", D](),
            grid_dim=D, block_dim=OLD_TPB,
        )


def new_col[B: Int, D: Int, ADT: DType](c: DeviceContext, mut b: Bufs) raises:
    comptime if ADT == DT:
        enqueue_bias_grad[DT](c, b.go.dev.value(), b.g1.dev.value(), B, D, b.part)
    else:
        enqueue_bias_grad[BF16](c, b.gob.dev.value(), b.g1.dev.value(), B, D, b.part)


def case_thread[B: Int, D: Int](c: DeviceContext, name: String) raises:
    var b = Bufs(c, B * D, D)
    run[old_thread[B, D], new_col[B, D, DT]](c, name, String("R=", B, " C=", D), b)


def case_col[B: Int, D: Int, ADT: DType](c: DeviceContext, name: String) raises:
    var b = Bufs(c, B * D, D)
    var dt = "bf16" if ADT == BF16 else "fp32"
    run[old_col[B, D, ADT], new_col[B, D, ADT]](
        c, name, String("R=", B, " C=", D, " ", dt), b
    )


# #7 LayerNormAct dγ/dβ with the recomputed Mish gate. `new` is the gate-free
# `enqueue_dot_bias_grad` (the same two sums), a proxy for the fixed kernel.
def old_lna[B: Int, D: Int](c: DeviceContext, mut b: Bufs) raises:
    c.enqueue_function[_old_ln_act_dparams[B, D, MishOp]](
        b.go.tt["gpu", B, D](), b.gamma.tt["gpu", D](), b.beta.tt["gpu", D](),
        b.x.tt["gpu", B, D](), b.g1.tt["gpu", D](), b.g2.tt["gpu", D](),
        grid_dim=D, block_dim=OLD_TPB,
    )


def new_lna[B: Int, D: Int](c: DeviceContext, mut b: Bufs) raises:
    enqueue_dot_bias_grad[DT](
        c, b.go.dev.value(), b.x.dev.value(), b.g1.dev.value(), b.g2.dev.value(),
        B, D, b.part,
    )


def case_lna[B: Int, D: Int](c: DeviceContext, name: String) raises:
    var b = Bufs(c, B * D, D)
    var t_old = 1e30
    var t_new = 1e30
    _ = _time_us[old_lna[B, D]](c, b)
    _ = _time_us[new_lna[B, D]](c, b)
    for _ in range(ROUNDS):
        t_old = min(t_old, _time_us[old_lna[B, D]](c, b))
        t_new = min(t_new, _time_us[new_lna[B, D]](c, b))
    var ratio = t_old / t_new
    var saved = t_old - t_new
    var verdict = "FIX" if ratio >= 1.3 and saved >= 20.0 else "not worth it"
    print(
        "|", name, "|", String("R=", B, " C=", D, " (proxy, no gate)"), "|",
        round(t_old, 1), "|", round(t_new, 1), "|", round(ratio, 2), "x |",
        round(saved, 1), "|", verdict, "| err n/a |",
    )


# #8 BatchNorm1D backward (train mode): old fused kernel vs zero + two-pass
# column sums + a coalesced dx pass (4 launches).
def case_bn1d[B: Int, D: Int](c: DeviceContext, name: String) raises:
    var b = Bufs(c, B * D, D)
    var gin = Tensor.alloc_gpu(c, B * D)
    var sg = Tensor.alloc_gpu(c, D)
    var sb = Tensor.alloc_gpu(c, D)
    var t_old = 1e30
    var t_new = 1e30
    for r in range(ROUNDS + 1):
        c.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            c.enqueue_function[_old_bn1d_bwd[B, D]](
                b.go.tt["gpu", B, D](), b.gamma.tt["gpu", D](),
                b.x.tt["gpu", B, D](), b.beta.tt["gpu", D](),
                gin.tt["gpu", B, D](), b.g1.tt["gpu", D](), b.g2.tt["gpu", D](),
                grid_dim=D, block_dim=OLD_TPB,
            )
        c.synchronize()
        var t1 = perf_counter_ns()
        for _ in range(N_CALLS):
            fill_dev(sg.dev.value(), D, Scalar[DT](0), c)
            fill_dev(sb.dev.value(), D, Scalar[DT](0), c)
            enqueue_dot_bias_grad[DT](
                c, b.go.dev.value(), b.x.dev.value(), sg.dev.value(),
                sb.dev.value(), B, D, b.part,
            )
            c.enqueue_function[_new_bn1d_dx[B, D]](
                b.go.tt["gpu", B, D](), b.gamma.tt["gpu", D](),
                b.x.tt["gpu", B, D](), b.beta.tt["gpu", D](),
                sg.tt["gpu", D](), sb.tt["gpu", D](),
                gin.tt["gpu", B, D](), b.g1.tt["gpu", D](), b.g2.tt["gpu", D](),
                grid_dim=(B * D + 255) // 256, block_dim=256,
            )
        c.synchronize()
        var t2 = perf_counter_ns()
        if r > 0:  # round 0 is the warm-up
            t_old = min(t_old, Float64(t1 - t0) / 1e3 / Float64(N_CALLS))
            t_new = min(t_new, Float64(t2 - t1) / 1e3 / Float64(N_CALLS))
    var ratio = t_old / t_new
    var saved = t_old - t_new
    var verdict = "FIX" if ratio >= 1.3 and saved >= 20.0 else "not worth it"
    print(
        "|", name, "|", String("R=", B, " C=", D, " (fused dx, 2 fills)"), "|",
        round(t_old, 1), "|", round(t_new, 1), "|", round(ratio, 2), "x |",
        round(saved, 1), "|", verdict, "| err n/a |",
    )


def main() raises:
    seed(7)
    var c = DeviceContext()
    print("param-grad reductions: old vs two-pass, us per call (best of", ROUNDS, "x", N_CALLS, ")")
    print("| kernel | shape | old us | new us | ratio | saved us | verdict | check |")
    # #1 Conv2DTranspose db — DreamerV3 CarRacing decoder, per step B=16.
    case_conv_db[16, 4, 96 * 96, DT, LAYOUT_NCHW](c, "#1 convT db (head)")
    case_conv_db[16, 48, 48 * 48, DT, LAYOUT_NCHW](c, "#1 convT db")
    case_conv_db[16, 96, 24 * 24, DT, LAYOUT_NCHW](c, "#1 convT db")
    case_conv_db[16, 192, 12 * 12, DT, LAYOUT_NCHW](c, "#1 convT db")
    # #2 Conv2D bf16 db — EZv2 atari tower (B=256, 6x6; NHWC default).
    case_conv_db[256, 16, 36, BF16, LAYOUT_NHWC](c, "#2 conv bf16 db")
    case_conv_db[256, 64, 36, BF16, LAYOUT_NHWC](c, "#2 conv bf16 db")
    case_conv_db[256, 64, 36, BF16, LAYOUT_NCHW](c, "#2 conv bf16 db")
    # #3 RMSNorm dγ — DreamerV3 WM (B=16: H=512, DETER=4096), DreamerV3
    # actor-critic heads (NS = B·T: Pong 1024 x 512 [50m] / 1024 [200m],
    # CarRacing 256 x 256), Dreamer4 (B·S=512, D=64).
    case_rms[16, 512](c, "#3 rms dgamma")
    case_rms[16, 4096](c, "#3 rms dgamma")
    case_rms[512, 64](c, "#3 rms dgamma")
    case_rms[256, 256](c, "#3 rms dgamma")
    case_rms[1024, 512](c, "#3 rms dgamma")
    case_rms[1024, 1024](c, "#3 rms dgamma")
    # Row sweep at C=512 (crossover of the two-pass path on this GPU).
    case_rms[64, 512](c, "sweep dot")
    case_rms[128, 512](c, "sweep dot")
    case_rms[256, 512](c, "sweep dot")
    case_rms[512, 512](c, "sweep dot")
    case_rms[2048, 512](c, "sweep dot")
    # #4 ConvRMSNorm dγ — DreamerV3 decoders, B=16.
    case_crms[16, 64, 48 * 48](c, "#4 conv rms dgamma")
    case_crms[16, 48, 48 * 48](c, "#4 conv rms dgamma")
    case_crms[16, 96, 24 * 24](c, "#4 conv rms dgamma")
    case_crms[16, 192, 12 * 12](c, "#4 conv rms dgamma")
    # #5 BlockLinear db — DreamerV3 RSSM, B=16.
    case_thread[16, 4096](c, "#5 block_linear db")
    case_thread[16, 12288](c, "#5 block_linear db")
    # #6 BiasAdd db — LeWM pos-emb (512 frames x 257·192), GPT-ish.
    case_col[512, 257 * 192, DT](c, "#6 bias_add db")
    case_col[512, 257 * 192, BF16](c, "#6 bias_add db")
    case_col[32, 256 * 384, DT](c, "#6 bias_add db")
    # #7 LayerNormAct dγ/dβ — TD-MPC2 (B=256, horizon-stacked 1024; DIM 512).
    case_lna[256, 512](c, "#7 ln_act dparams")
    case_lna[1024, 512](c, "#7 ln_act dparams")
    # #9 LSTM db (EZv2 B=256, 4H=2048), Noisy b (DQN B=128, OUT 512).
    case_col[256, 2048, DT](c, "#9 lstm db")
    case_col[128, 512, DT](c, "#9 noisy b")
    # #8 BN1D backward — LeWM projector (512 frames x HID 2048), EZv2
    # projector (B=256 x PROJ_HID) and LSTM-head BN (B=256 x 512).
    case_bn1d[512, 2048](c, "#8 bn1d bwd")
    case_bn1d[256, 1024](c, "#8 bn1d bwd")
    case_bn1d[256, 512](c, "#8 bn1d bwd")
