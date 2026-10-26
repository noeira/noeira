"""ConvTransposePCBlock validation — adjoint + finite-difference (CPU).

A transposed conv is the adjoint of a forward conv, so we validate without a
separate oracle implementation:

  1. Adjoint identity: with ACT=Identity, bias=0, predict is a linear map
     T: x_small → μ_big and pull_back is P: ε_big → z_small. They must satisfy
     ⟨T x, y⟩_big == ⟨x, P y⟩_small for random x, y.
  2. weight_grad must equal dE/dW for E = ½‖x_above − μ(W)‖² (finite difference).

Run:
    pixi run mojo run -I . tests/pcn/test_conv_transpose_block.mojo
"""

from std.memory import alloc
from std.math import sin
from noeira.core.tiles import Tile, tile

from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_conv_transpose_block import (
    ConvTransposePCBlock,
)
from noeira.experimental.pcn import PCIdentity

comptime dtype = DType.float32

# 3×3 (small) → 6×6 (big) upsample, kernel 4 / stride 2 / pad 1.
comptime CB = ConvTransposePCBlock[2, 3, 4, 2, 1, 3, 3, PCIdentity]
comptime BATCH = 2
comptime IN = CB.IN_DIM
comptime OUT = CB.OUT_DIM
comptime PSZ = CB.PARAM_SIZE


def _energy(
    x_small: Tile[dtype, BATCH, IN],
    x_above: Pointer[Scalar[dtype], MutAnyOrigin],
    params: Tile[dtype, PSZ],
    mu: Tile[dtype, BATCH, OUT],
    a: Tile[dtype, BATCH, IN],
) -> Float64:
    CB.predict[BATCH, dtype](x_small, params, mu, a)
    var e: Float64 = 0.0
    for i in range(BATCH * OUT):
        var d = Float64(x_above[i]) - Float64(mu.ptr[i])
        e += 0.5 * d * d
    return e


def main() raises:
    print("ConvTransposePCBlock validation (adjoint + FD)\n")
    print("  small=", CB.in_h, "x", CB.in_w, " big=", CB.out_h, "x", CB.out_w,
          " IN=", IN, " OUT=", OUT, " PARAM_SIZE=", PSZ)

    var x_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var y_buf = alloc[Scalar[dtype]](BATCH * OUT).as_unsafe_any_origin()
    var params_buf = alloc[Scalar[dtype]](PSZ).as_unsafe_any_origin()
    var mu_buf = alloc[Scalar[dtype]](BATCH * OUT).as_unsafe_any_origin()
    var a_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var z_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()

    for i in range(BATCH * IN):
        x_buf[i] = Scalar[dtype](sin(Float32(i) * 0.7 + 0.2))
    for i in range(BATCH * OUT):
        y_buf[i] = Scalar[dtype](sin(Float32(i) * 1.3 + 0.9))

    var x = tile[BATCH, IN](x_buf)
    var y = tile[BATCH, OUT](y_buf)
    var params = tile[PSZ](params_buf)
    var mu = tile[BATCH, OUT](mu_buf)
    var a = tile[BATCH, IN](a_buf)
    var z = tile[BATCH, IN](z_buf)
    CB.pc_init_params[PCXavier, dtype](params)  # bias starts at 0

    # ── 1. Adjoint identity ───────────────────────────────────────────────────
    CB.predict[BATCH, dtype](x, params, mu, a)  # μ = T x  (Identity act, 0 bias)
    CB.pull_back[BATCH, dtype](y, params, z)  # z = P y
    var lhs: Float64 = 0.0
    for i in range(BATCH * OUT):
        lhs += Float64(mu_buf[i]) * Float64(y_buf[i])
    var rhs: Float64 = 0.0
    for i in range(BATCH * IN):
        rhs += Float64(x_buf[i]) * Float64(z_buf[i])
    var adj_rel = abs(lhs - rhs) / (abs(lhs) + 1e-9)
    print("\n  [adjoint]  <Tx,y> =", lhs, "  <x,Py> =", rhs,
          "  rel=", adj_rel)
    var adjoint_ok = adj_rel < 1e-4

    # ── 2. Finite-difference weight_grad ─────────────────────────────────────
    # E = ½‖x_above − μ‖² with x_above = y; grads must equal dE/dW.
    CB.predict[BATCH, dtype](x, params, mu, a)
    var eps_buf = alloc[Scalar[dtype]](BATCH * OUT).as_unsafe_any_origin()
    for i in range(BATCH * OUT):
        eps_buf[i] = y_buf[i] - mu_buf[i]
    var eps = tile[BATCH, OUT](eps_buf)
    var grads_buf = alloc[Scalar[dtype]](PSZ).as_unsafe_any_origin()
    var grads = tile[PSZ](grads_buf)
    CB.weight_grad[BATCH, dtype](eps, a, grads)

    var h: Float64 = 1e-2
    var max_rel: Float64 = 0.0
    var worst_i: Int = 0
    for i in range(PSZ):
        var saved = params_buf[i]
        params_buf[i] = Scalar[dtype](Float64(saved) + h)
        var e_plus = _energy(x, y_buf, params, mu, a)
        params_buf[i] = Scalar[dtype](Float64(saved) - h)
        var e_minus = _energy(x, y_buf, params, mu, a)
        params_buf[i] = saved
        var g_fd = (e_plus - e_minus) / (2.0 * h)
        var g_an = Float64(grads_buf[i])
        var denom = abs(g_fd) + abs(g_an) + 1e-6
        var rel = abs(g_fd - g_an) / denom
        if rel > max_rel:
            max_rel = rel
            worst_i = i
    print("  [FD wgrad] max rel error =", max_rel, " at idx", worst_i,
          " (analytic=", Float64(grads_buf[worst_i]), ")")
    var fd_ok = max_rel < 1e-2

    x_buf.free()
    y_buf.free()
    params_buf.free()
    mu_buf.free()
    a_buf.free()
    z_buf.free()
    eps_buf.free()
    grads_buf.free()

    print("")
    if adjoint_ok and fd_ok:
        print("✅ PASS — ConvTransposePCBlock predict/pull_back are adjoints",
              "and weight_grad matches FD")
    else:
        if not adjoint_ok:
            print("❌ FAIL — adjoint identity violated")
        if not fd_ok:
            print("❌ FAIL — weight_grad disagrees with finite difference")
        raise Error("ConvTransposePCBlock validation failed")
