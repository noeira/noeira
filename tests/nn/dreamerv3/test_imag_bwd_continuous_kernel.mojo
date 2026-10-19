"""Unit parity for `_imag_bwd_continuous_k` — the device continuous imag-loss
forward(loss)+backward(grads) kernel — vs the host `imag_loss_cpu` /
`imag_loss_backward` (DISCRETE=False).

This is the hardest, parity-critical kernel of the continuous-AC device port:
the bounded-normal log-prob/entropy + chain-rule backward. Feed identical fixed
histories (vlog/svlog/pmean/pstd/acts/conv/bins) + a precomputed ret/rscale, run
host loss+backward (cotangent d_policy=d_value=inv_im) and the device kernel,
compare polloss/valloss + grad_vlogits/grad_pmean/grad_pstd_raw.

Run: pixi run -e apple mojo run -I . tests/nn/dreamerv3/test_imag_bwd_continuous_kernel.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.testing import assert_true

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.deep_agents.dreamerv3.blocks import _imag_bwd_continuous_k
from noeira.deep_agents.dreamerv3.imag_loss import (
    imag_loss_cpu, imag_loss_backward,
)
from noeira.deep_agents.dreamerv3.normalize import PercentileNormalize

comptime NS = 6
comptime TI = 4
comptime ACT = 3
comptime BINS = 7
comptime TM1 = TI - 1
comptime MINSTD = Scalar[DT](0.1)
comptime MAXSTD = Scalar[DT](1.0)
comptime ACTENT = Scalar[DT](3e-4)
comptime SLOWREG = Scalar[DT](1.0)
comptime LAM = Scalar[DT](0.95)
comptime INV_IM = Scalar[DT](1.0)


def _hp(t: Tensor) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return rebind[Pointer[Scalar[DT], MutAnyOrigin]](t.data.unsafe_ptr())


def main() raises:
    print("--- _imag_bwd_continuous_k device-vs-host parity ---")
    var ctx = DeviceContext()

    # ── fixed inputs ──
    var vlog = Tensor.alloc(NS * TI * BINS)
    var svlog = Tensor.alloc(NS * TI * BINS)
    var pmean = Tensor.alloc(NS * TI * ACT)
    var pstd = Tensor.alloc(NS * TI * ACT)
    var acts = Tensor.alloc(NS * TI * ACT)
    var con = Tensor.alloc(NS * TI)
    var rew = Tensor.alloc(NS * TI)
    var bins = Tensor.alloc(BINS)
    for i in range(NS * TI * BINS):
        vlog.data[i] = Scalar[DT]((i * 7 + 3) % 13 - 6) * 0.11
        svlog.data[i] = Scalar[DT]((i * 5 + 2) % 11 - 5) * 0.09
    for i in range(NS * TI * ACT):
        pmean.data[i] = Scalar[DT]((i * 3 + 1) % 9 - 4) * 0.2
        pstd.data[i] = Scalar[DT]((i * 2 + 5) % 7 - 3) * 0.3
        acts.data[i] = Scalar[DT]((i * 11 + 4) % 5 - 2) * 0.25
    for i in range(NS * TI):
        con.data[i] = Scalar[DT](0.9) + Scalar[DT](i % 3) * 0.03
        rew.data[i] = Scalar[DT]((i * 5 + 1) % 7 - 3) * 0.2
    for c in range(BINS):
        bins.data[c] = Scalar[DT](-6.0 + 12.0 * Float64(c) / Float64(BINS - 1))

    # ── host forward (loss + ret + rscale) ──
    var out_pol = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
    var out_val = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
    var out_ret = List[Scalar[DT]](length=NS * TM1, fill=Scalar[DT](0))
    var rn = PercentileNormalize.make("perc", debias=False)
    imag_loss_cpu[NS, TI, ACT, BINS, False](
        _hp(acts), _hp(rew), _hp(con), _hp(vlog), _hp(svlog), _hp(pmean),
        _hp(pstd), _hp(bins), MINSTD, MAXSTD, LAM, ACTENT, SLOWREG, rn,
        out_pol, out_val, out_ret, slowtar=False,
    )
    var rscale = rn.stats()[1]

    # ── host backward (grads), cotangents = INV_IM ──
    var gvlog_h = Tensor.alloc(NS * TI * BINS)
    var gpmean_h = Tensor.alloc(NS * TI * ACT)
    var gpstd_h = Tensor.alloc(NS * TI * ACT)
    var d_pol = Tensor.alloc(NS * TM1)
    var d_val = Tensor.alloc(NS * TM1)
    for i in range(NS * TM1):
        d_pol.data[i] = INV_IM
        d_val.data[i] = INV_IM
    imag_loss_backward[NS, TI, ACT, BINS, False](
        _hp(acts), _hp(rew), _hp(con), _hp(vlog), _hp(svlog), _hp(pmean),
        _hp(pstd), _hp(bins), MINSTD, MAXSTD, LAM, ACTENT, SLOWREG, rscale,
        _hp(d_pol), _hp(d_val),
        _hp(gvlog_h), _hp(gpmean_h), _hp(gpstd_h), slowtar=False,
    )

    # ── device kernel ──
    var ret_d = Tensor.alloc(NS * TM1)
    for i in range(NS * TM1):
        ret_d.data[i] = out_ret[i]
    var rscale_t = Tensor.alloc(1)
    rscale_t.data[0] = rscale
    vlog.ensure_gpu(ctx, NS * TI * BINS); vlog.upload(ctx)
    svlog.ensure_gpu(ctx, NS * TI * BINS); svlog.upload(ctx)
    pmean.ensure_gpu(ctx, NS * TI * ACT); pmean.upload(ctx)
    pstd.ensure_gpu(ctx, NS * TI * ACT); pstd.upload(ctx)
    acts.ensure_gpu(ctx, NS * TI * ACT); acts.upload(ctx)
    con.ensure_gpu(ctx, NS * TI); con.upload(ctx)
    bins.ensure_gpu(ctx, BINS); bins.upload(ctx)
    ret_d.ensure_gpu(ctx, NS * TM1); ret_d.upload(ctx)
    rscale_t.ensure_gpu(ctx, 1); rscale_t.upload(ctx)

    var gvlog_d = Tensor(); gvlog_d.ensure_gpu(ctx, NS * TI * BINS)
    var gpmean_d = Tensor(); gpmean_d.ensure_gpu(ctx, NS * TI * ACT)
    var gpstd_d = Tensor(); gpstd_d.ensure_gpu(ctx, NS * TI * ACT)
    var pol_d = Tensor(); pol_d.ensure_gpu(ctx, NS * TM1)
    var val_d = Tensor(); val_d.ensure_gpu(ctx, NS * TM1)

    comptime nb = (NS + TPB - 1) // TPB
    ctx.enqueue_function[_imag_bwd_continuous_k[NS, TI, BINS, ACT]](
        vlog.tt["gpu", NS * TI * BINS](),
        svlog.tt["gpu", NS * TI * BINS](),
        pmean.tt["gpu", NS * TI * ACT](),
        pstd.tt["gpu", NS * TI * ACT](),
        acts.tt["gpu", NS * TI * ACT](),
        con.tt["gpu", NS * TI](),
        ret_d.tt["gpu", NS * TM1](),
        bins.tt["gpu", BINS](),
        gvlog_d.tt["gpu", NS * TI * BINS](),
        gpmean_d.tt["gpu", NS * TI * ACT](),
        gpstd_d.tt["gpu", NS * TI * ACT](),
        pol_d.tt["gpu", NS * TM1](),
        val_d.tt["gpu", NS * TM1](),
        rscale_t.tt["gpu", 1](),
        ACTENT, SLOWREG, INV_IM, MINSTD, MAXSTD,
        grid_dim=nb, block_dim=TPB,
    )
    gvlog_d.download(ctx)
    gpmean_d.download(ctx)
    gpstd_d.download(ctx)
    pol_d.download(ctx)
    val_d.download(ctx)
    ctx.synchronize()

    var d_pl: Float64 = 0.0
    var d_vl: Float64 = 0.0
    for i in range(NS * TM1):
        d_pl = max(d_pl, abs(Float64(pol_d.data[i] - out_pol[i])))
        d_vl = max(d_vl, abs(Float64(val_d.data[i] - out_val[i])))
    var d_gv: Float64 = 0.0
    for i in range(NS * TI * BINS):
        d_gv = max(d_gv, abs(Float64(gvlog_d.data[i] - gvlog_h.data[i])))
    var d_gm: Float64 = 0.0
    var d_gs: Float64 = 0.0
    for i in range(NS * TI * ACT):
        d_gm = max(d_gm, abs(Float64(gpmean_d.data[i] - gpmean_h.data[i])))
        d_gs = max(d_gs, abs(Float64(gpstd_d.data[i] - gpstd_h.data[i])))
    print("  max|Δ|: polloss", d_pl, " valloss", d_vl, " gvlog", d_gv,
          " gpmean", d_gm, " gpstd", d_gs)
    var tol = 1e-4
    assert_true(
        d_pl < tol and d_vl < tol and d_gv < tol and d_gm < tol and d_gs < tol,
        "continuous imag-bwd kernel matches host imag_loss",
    )
    print("IMAG_BWD_CONTINUOUS KERNEL PARITY PASSED")
