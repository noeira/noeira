"""Unit parity for `_gaussian_sample_hist_k` (continuous AC device sampler).

The device Gaussian sampler must match the host bounded-normal sample
(`tanh(mean_raw) + bounded_std(std_raw)·noise`) bit-for-bit — it's the
foundational kernel of the continuous-AC device-residency port (Phase 1). Feed
random policy outputs [NS,2·ACT] + fixed noise, run the kernel over TI steps,
and compare acts/pmean/pstd histories + the per-step `at` to the host math.

Run: pixi run -e apple mojo run -I . tests/nn/dreamerv3/test_gaussian_sample_kernel.mojo
"""

from std.math import tanh, exp, abs
from max.gpu.host import DeviceContext
from std.testing import assert_true

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.deep_agents.dreamerv3.blocks import _gaussian_sample_hist_k
from noeira.deep_agents.dreamerv3.dists import bounded_std

comptime NS = 5
comptime TI = 3
comptime ACT = 4
comptime MINSTD = Scalar[DT](0.1)
comptime MAXSTD = Scalar[DT](1.0)


def main() raises:
    print("--- _gaussian_sample_hist_k device-vs-host parity ---")
    var ctx = DeviceContext()

    # random-ish policy outputs [NS, 2*ACT] and noise [TI, NS, ACT]
    var pb = Tensor.alloc(NS * 2 * ACT)
    for i in range(NS * 2 * ACT):
        pb.data[i] = Scalar[DT]((i * 7 + 3) % 11 - 5) * 0.21
    var noise = Tensor.alloc(TI * NS * ACT)
    for i in range(TI * NS * ACT):
        noise.data[i] = Scalar[DT]((i * 5 + 1) % 9 - 4) * 0.17
    pb.ensure_gpu(ctx, NS * 2 * ACT)
    pb.upload(ctx)
    noise.ensure_gpu(ctx, TI * NS * ACT)
    noise.upload(ctx)

    var at = Tensor()
    at.ensure_gpu(ctx, NS * ACT)
    var pmean = Tensor()
    pmean.ensure_gpu(ctx, NS * TI * ACT)
    var pstd = Tensor()
    pstd.ensure_gpu(ctx, NS * TI * ACT)
    var acts = Tensor()
    acts.ensure_gpu(ctx, NS * TI * ACT)

    comptime nb = (NS + TPB - 1) // TPB
    for t in range(TI):
        ctx.enqueue_function[_gaussian_sample_hist_k[ACT, TI, NS]](
            pb.tt["gpu", NS * 2 * ACT](),
            noise.tt["gpu", TI * NS * ACT](),
            at.tt["gpu", NS * ACT](),
            pmean.tt["gpu", NS * TI * ACT](),
            pstd.tt["gpu", NS * TI * ACT](),
            acts.tt["gpu", NS * TI * ACT](),
            MINSTD,
            MAXSTD,
            Int64(t),
            grid_dim=nb,
            block_dim=TPB,
        )
    at.download(ctx)
    pmean.download(ctx)
    pstd.download(ctx)
    acts.download(ctx)
    ctx.synchronize()

    var d_acts: Float64 = 0.0
    var d_pm: Float64 = 0.0
    var d_ps: Float64 = 0.0
    var d_at: Float64 = 0.0
    for b in range(NS):
        for t in range(TI):
            for a in range(ACT):
                var mr = pb.data[b * 2 * ACT + a]
                var sr = pb.data[b * 2 * ACT + ACT + a]
                var z = noise.data[(t * NS + b) * ACT + a]
                var ref_smp = tanh(mr) + bounded_std(sr, MINSTD, MAXSTD) * z
                var idx = (b * TI + t) * ACT + a
                d_acts = max(d_acts, abs(Float64(acts.data[idx] - ref_smp)))
                d_pm = max(d_pm, abs(Float64(pmean.data[idx] - mr)))
                d_ps = max(d_ps, abs(Float64(pstd.data[idx] - sr)))
                # `at` holds the LAST t's sample
                if t == TI - 1:
                    d_at = max(
                        d_at, abs(Float64(at.data[b * ACT + a] - ref_smp))
                    )
    print("  max|Δ|: acts", d_acts, " pmean", d_pm, " pstd", d_ps, " at", d_at)
    var tol = 1e-6
    assert_true(
        d_acts < tol and d_pm < tol and d_ps < tol and d_at < tol,
        "gaussian sample kernel matches host bounded-normal sample",
    )
    print("GAUSSIAN SAMPLE KERNEL PARITY PASSED")
