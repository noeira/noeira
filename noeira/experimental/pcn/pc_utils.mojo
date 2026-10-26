"""Shared utilities for pcn training loops.

Promoted from amortized PC test files
(`test_{pendulum,mountain_car}_amortized_pc.mojo`).
"""

from std.math import sqrt
from std.sys import simd_width_of
from noeira.core.tiles import Tile

comptime _SW = simd_width_of[DType.float32]()


def clip_grad_norm[
    SIZE: Int, dtype: DType = DType.float32
](
    grads: Tile[dtype, SIZE],
    max_norm: Float64,
):
    """Global L2-norm gradient clipping (in-place).

    Computes `||grads||_2`. If it exceeds `max_norm`, scales every element
    by `max_norm / ||grads||_2`. Reduces with Float64 accumulation to keep
    the norm stable when `dtype` is Float32.

    No-op when the norm is already within budget.
    """
    var gp = grads.ptr
    var sum_sq: Float64 = 0
    comptime if dtype == DType.float32:
        var acc = SIMD[dtype, _SW](0)
        var i = 0
        while i + _SW <= SIZE:
            var v = gp.unsafe_load[width=_SW](i)
            acc = acc + v * v
            i += _SW
        sum_sq = Float64(acc.reduce_add())
        while i < SIZE:
            var g = Float64(gp[unsafe_offset=i])
            sum_sq += g * g
            i += 1
    else:
        for i in range(SIZE):
            var g = Float64(gp[unsafe_offset=i])
            sum_sq += g * g
    var norm = sqrt(sum_sq)
    if norm > max_norm:
        var scale = Scalar[dtype](max_norm / norm)
        comptime if dtype == DType.float32:
            var sv = SIMD[dtype, _SW](scale)
            var i = 0
            while i + _SW <= SIZE:
                gp.unsafe_store(i, gp.unsafe_load[width=_SW](i) * sv)
                i += _SW
            while i < SIZE:
                gp[unsafe_offset=i] = gp[unsafe_offset=i] * scale
                i += 1
        else:
            for i in range(SIZE):
                gp[unsafe_offset=i] = gp[unsafe_offset=i] * scale


def spectral_norm_clamp[
    IN: Int, OUT: Int, dtype: DType = DType.float32
](
    W: Tile[dtype, IN, OUT],
    u: Tile[dtype, IN],
    v: Tile[dtype, OUT],
    target_sigma: Float64,
    n_power_iters: Int = 1,
) -> Float64:
    """In-place spectral-norm projection (Miyato 2018, power iteration).

    Estimates the largest singular value σ_max(W) by `n_power_iters` rounds of
    power iteration starting from the persistent (u, v) estimate. If σ_max
    exceeds `target_sigma`, rescales every element of W by
    `target_sigma / σ_max` so the projected matrix has σ_max ≤ `target_sigma`.

    Persistent (u, v) means the caller stores them across calls — one power
    iteration per call is enough to track σ_max if it changes slowly (it does
    during gradient descent). Initialize (u, v) once with non-zero values
    (e.g., Gaussian noise) before the first call.

    Returns the σ_max estimate from this call (after the power iterations,
    before any rescaling). Use it to log spectral radius drift.

    Float64 reductions for stability when `dtype` is Float32. Skips clamping
    if the iteration produces a degenerate (zero-norm) singular vector.
    """
    var u_buf = List[Float64](capacity=IN)
    var v_buf = List[Float64](capacity=OUT)
    for i in range(IN):
        u_buf.append(Float64(u.ptr[unsafe_offset=i]))
    for j in range(OUT):
        v_buf.append(Float64(v.ptr[unsafe_offset=j]))

    for _ in range(n_power_iters):
        # v = W^T u; normalize.
        var v_norm_sq: Float64 = 0
        for j in range(OUT):
            var s: Float64 = 0
            for i in range(IN):
                s += u_buf[i] * Float64(W.ptr[unsafe_offset=i * OUT + j])
            v_buf[j] = s
            v_norm_sq += s * s
        var v_norm = sqrt(v_norm_sq)
        if v_norm < 1e-12:
            return 0.0
        for j in range(OUT):
            v_buf[j] = v_buf[j] / v_norm

        # u = W v; normalize.
        var u_norm_sq: Float64 = 0
        for i in range(IN):
            var s: Float64 = 0
            for j in range(OUT):
                s += Float64(W.ptr[unsafe_offset=i * OUT + j]) * v_buf[j]
            u_buf[i] = s
            u_norm_sq += s * s
        var u_norm = sqrt(u_norm_sq)
        if u_norm < 1e-12:
            return 0.0
        for i in range(IN):
            u_buf[i] = u_buf[i] / u_norm

    # Persist refined (u, v).
    for i in range(IN):
        u.ptr[unsafe_offset=i] = Scalar[dtype](u_buf[i])
    for j in range(OUT):
        v.ptr[unsafe_offset=j] = Scalar[dtype](v_buf[j])

    # σ ≈ u^T W v.
    var sigma: Float64 = 0
    for i in range(IN):
        var s: Float64 = 0
        for j in range(OUT):
            s += Float64(W.ptr[unsafe_offset=i * OUT + j]) * v_buf[j]
        sigma += u_buf[i] * s

    if sigma > target_sigma:
        var scale = Scalar[dtype](target_sigma / sigma)
        comptime N = IN * OUT
        comptime if dtype == DType.float32:
            var sv = SIMD[dtype, _SW](scale)
            var k = 0
            while k + _SW <= N:
                W.ptr.unsafe_store(k, W.ptr.unsafe_load[width=_SW](k) * sv)
                k += _SW
            while k < N:
                W.ptr[unsafe_offset=k] = W.ptr[unsafe_offset=k] * scale
                k += 1
        else:
            for k in range(N):
                W.ptr[unsafe_offset=k] = W.ptr[unsafe_offset=k] * scale

    return sigma
