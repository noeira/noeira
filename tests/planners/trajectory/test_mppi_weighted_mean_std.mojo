"""`mppi_weighted_mean_std_kernel` must match a host reference exactly enough.

The kernel refits the MPPI proposal distribution, so a wrong reduction does not
crash — it just plans worse. It was rewritten from one-thread-per-output (which
could only ever launch 1 BLOCK, because there are fewer outputs than threads)
to one-BLOCK-per-output with an intra-block tree reduction. This gate recomputes
both `mean` and `std` on the host from the same weights/actions and compares.

⚠ THE BAR IS MIXED (atol + rtol*|ref|), NOT PURE RELATIVE. The weighted mean
is a CANCELLING sum — summands are ~1.4e-3 and some outputs land near 1e-5 —
so fp32 reduction-order noise of ~1.6e-9 ABSOLUTE reads as 2e-4 RELATIVE and a
pure relative bar rejects a correct kernel. The std, summing only positive
terms, has no cancellation and matches to 7.9e-8 relative. atol=1e-7 is ~60x
the observed noise and ~1e4x below the summand scale; a mis-indexed reduction
produces O(1e-3) absolute error and fails by four orders of magnitude.

    pixi run -e apple mojo run -I . tests/planners/trajectory/test_mppi_weighted_mean_std.mojo
"""

from std.math import abs, sqrt
from max.gpu.host import DeviceContext
from noeira.core.tiles import tile

from noeira.nn.constants import DT
from noeira.planners.trajectory.mppi_kernels import (
    mppi_weighted_mean_std_kernel,
)

comptime TPB = 256
comptime N_ENVS = 8
comptime TOTAL_SAMPLES = 268
comptime HORIZON = 3
comptime ACT = 6
comptime DIMS = N_ENVS * HORIZON * ACT          # 144 — FEWER than one block
comptime NW = N_ENVS * TOTAL_SAMPLES
comptime NA = N_ENVS * TOTAL_SAMPLES * HORIZON * ACT


def main() raises:
    var ctx = DeviceContext()
    print("MPPI weighted_mean_std —", ctx.name())
    print(
        "  outputs=", DIMS, " samples/output=", TOTAL_SAMPLES,
        "   old grid = ", (DIMS + TPB - 1) // TPB, " block(s);  new grid = ",
        DIMS, " blocks", sep="",
    )

    var wh = ctx.enqueue_create_host_buffer[DT](NW)
    var ah = ctx.enqueue_create_host_buffer[DT](NA)
    # Non-uniform, non-degenerate weights: normalized per env so the reference
    # mean is a true weighted mean.
    for e in range(N_ENVS):
        var tot = Float64(0)
        for s in range(TOTAL_SAMPLES):
            var v = 0.1 + Float64((s * 37 + e * 11) % 53) / 53.0
            wh[e * TOTAL_SAMPLES + s] = Scalar[DT](v)
            tot += v
        for s in range(TOTAL_SAMPLES):
            wh[e * TOTAL_SAMPLES + s] = Scalar[DT](
                Float64(wh[e * TOTAL_SAMPLES + s]) / tot
            )
    for i in range(NA):
        ah[i] = Scalar[DT](0.01) * Scalar[DT]((i % 71) - 35)

    var wd = ctx.enqueue_create_buffer[DT](NW)
    var ad = ctx.enqueue_create_buffer[DT](NA)
    var md = ctx.enqueue_create_buffer[DT](DIMS)
    var sd = ctx.enqueue_create_buffer[DT](DIMS)
    ctx.enqueue_copy(wd, wh)
    ctx.enqueue_copy(ad, ah)

    comptime k = mppi_weighted_mean_std_kernel[
        DT, N_ENVS, TOTAL_SAMPLES, HORIZON, ACT, TPB
    ]
    ctx.enqueue_function[k](
        tile[NW](wd),
        tile[NA](ad),
        tile[DIMS](md),
        tile[DIMS](sd),
        grid_dim=(DIMS,),
        block_dim=(TPB,),
    )
    var mo = ctx.enqueue_create_host_buffer[DT](DIMS)
    var so = ctx.enqueue_create_host_buffer[DT](DIMS)
    ctx.enqueue_copy(mo, md)
    ctx.enqueue_copy(so, sd)
    ctx.synchronize()

    var max_abs_m = Float64(0)
    var max_rel_m = Float64(0)
    var max_rel_s = Float64(0)
    var mag = Float64(0)
    for o in range(DIMS):
        var e = o // (HORIZON * ACT)
        var rem = o % (HORIZON * ACT)
        var t = rem // ACT
        var a = rem % ACT
        var woff = e * TOTAL_SAMPLES
        var aoff = e * TOTAL_SAMPLES * HORIZON * ACT + t * ACT + a
        var m_ref = Float64(0)
        for s in range(TOTAL_SAMPLES):
            m_ref += Float64(wh[woff + s]) * Float64(
                ah[aoff + s * HORIZON * ACT]
            )
        var v_ref = Float64(0)
        for s in range(TOTAL_SAMPLES):
            var d = Float64(ah[aoff + s * HORIZON * ACT]) - m_ref
            v_ref += Float64(wh[woff + s]) * d * d
        var s_ref = sqrt(v_ref + 1e-8)
        if s_ref < 0.05:
            s_ref = 0.05
        if s_ref > 2.0:
            s_ref = 2.0

        if abs(m_ref) > mag:
            mag = abs(m_ref)
        # mixed tolerance: |err| <= atol + rtol*|ref|  (see the header)
        var em = abs(Float64(mo[o]) - m_ref)
        var es = abs(Float64(so[o]) - s_ref)
        if em > max_abs_m:
            max_abs_m = em
        var dm = em / (1e-7 + 1e-5 * abs(m_ref))
        var ds = es / (1e-7 + 1e-5 * abs(s_ref))
        if dm > max_rel_m:
            max_rel_m = dm
        if ds > max_rel_s:
            max_rel_s = ds

    print(
        "  mean err/tol=", max_rel_m, " (max_abs=", max_abs_m,
        ", |mean|max=", mag, ")   std err/tol=", max_rel_s, sep="",
    )
    # ⚠ NON-VACUITY: an all-zero mean would compare equal to a broken kernel.
    if mag == 0.0:
        raise Error("VACUOUS: the reference mean is identically zero")
    # err/tol > 1 means the mixed tolerance was exceeded.
    if max_rel_m > 1.0 or max_rel_s > 1.0:
        raise Error(
            "weighted_mean_std mismatch: mean err/tol " + String(max_rel_m)
            + "  std err/tol " + String(max_rel_s)
        )
    print("PASSED")
