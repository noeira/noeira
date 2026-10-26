"""NormConvPCBlock validation — finite differences (CPU).

Validates the fused conv + input-side per-channel RMSNorm block (no separate
norm level). Transform μ = Conv(RMSNorm_ch(ACT(x_below))). FD checks:
  1. pull_back → act_derivative_mul must equal J^T ε (numerical via predict).
  2. weight_grad must equal dE/dW for E = ½‖x_above − μ‖² (conv weights+bias).

ACT=PCReLU with positive-biased inputs (ReLU active, no kinks → clean FD).

Run:
    pixi run mojo run -I . tests/pcn/test_norm_conv_pc_block.mojo
"""

from std.memory import alloc
from std.math import sin
from noeira.core.tiles import Tile, tile

from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_norm_conv_block import NormConvPCBlock
from noeira.experimental.pcn import PCReLU

comptime dtype = DType.float32
comptime CB = NormConvPCBlock[2, 3, 3, 1, 1, 4, 4, PCReLU]
comptime BATCH = 2
comptime IN = CB.IN_DIM
comptime OUT = CB.OUT_DIM
comptime PSZ = CB.PARAM_SIZE


def _predict(
    x: Tile[dtype, BATCH, IN],
    params: Tile[dtype, PSZ],
    mu: Tile[dtype, BATCH, OUT],
    a: Tile[dtype, BATCH, IN],
):
    CB.predict[BATCH, dtype](x, params, mu, a)


def main() raises:
    print("NormConvPCBlock validation (finite differences)\n")
    print("  IN=", IN, " OUT=", OUT, " PARAM_SIZE=", PSZ)

    var x_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var y_buf = alloc[Scalar[dtype]](BATCH * OUT).as_unsafe_any_origin()
    var params_buf = alloc[Scalar[dtype]](PSZ).as_unsafe_any_origin()
    var mu_buf = alloc[Scalar[dtype]](BATCH * OUT).as_unsafe_any_origin()
    var a_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var eps_buf = alloc[Scalar[dtype]](BATCH * OUT).as_unsafe_any_origin()
    var z_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var zeff_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](PSZ).as_unsafe_any_origin()

    for i in range(BATCH * IN):
        x_buf[i] = Scalar[dtype](sin(Float32(i) * 0.7 + 0.3) * 0.5 + 1.0)  # >0
    for i in range(BATCH * OUT):
        y_buf[i] = Scalar[dtype](sin(Float32(i) * 1.1 + 0.6))

    var x = tile[BATCH, IN](x_buf)
    var y = tile[BATCH, OUT](y_buf)
    var params = tile[PSZ](params_buf)
    var mu = tile[BATCH, OUT](mu_buf)
    var a = tile[BATCH, IN](a_buf)
    var eps = tile[BATCH, OUT](eps_buf)
    var z = tile[BATCH, IN](z_buf)
    var zeff = tile[BATCH, IN](zeff_buf)
    var grads = tile[PSZ](grads_buf)
    CB.pc_init_params[PCXavier, dtype](params)

    _predict(x, params, mu, a)
    CB.eps_compute[BATCH, dtype](y, mu, eps)
    CB.pull_back[BATCH, dtype](eps, params, z)
    CB.act_derivative_mul[BATCH, dtype](x, z, zeff)

    var h: Float64 = 1e-3
    var mu_p = alloc[Scalar[dtype]](BATCH * OUT).as_unsafe_any_origin()
    var mu_m = alloc[Scalar[dtype]](BATCH * OUT).as_unsafe_any_origin()
    var a_tmp = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var mu_p_t = tile[BATCH, OUT](mu_p)
    var mu_m_t = tile[BATCH, OUT](mu_m)
    var a_tmp_t = tile[BATCH, IN](a_tmp)

    var max_jvp_rel: Float64 = 0.0
    for b in range(BATCH):
        for j in range(IN):
            var saved = x_buf[b * IN + j]
            x_buf[b * IN + j] = Scalar[dtype](Float64(saved) + h)
            _predict(x, params, mu_p_t, a_tmp_t)
            x_buf[b * IN + j] = Scalar[dtype](Float64(saved) - h)
            _predict(x, params, mu_m_t, a_tmp_t)
            x_buf[b * IN + j] = saved
            var jtv: Float64 = 0.0
            for i in range(OUT):
                var dmu = (Float64(mu_p[b * OUT + i]) - Float64(mu_m[b * OUT + i])) / (2.0 * h)
                jtv += Float64(eps_buf[b * OUT + i]) * dmu
            var an = Float64(zeff_buf[b * IN + j])
            var rel = abs(jtv - an) / (abs(jtv) + abs(an) + 1e-6)
            if rel > max_jvp_rel:
                max_jvp_rel = rel
    print("\n  [pull_back∘act_deriv vs J^Tε]  max rel error =", max_jvp_rel)
    var jvp_ok = max_jvp_rel < 5e-2

    CB.weight_grad[BATCH, dtype](eps, a, grads)
    var max_g_rel: Float64 = 0.0
    for i in range(PSZ):
        var saved = params_buf[i]
        params_buf[i] = Scalar[dtype](Float64(saved) + h)
        _predict(x, params, mu_p_t, a_tmp_t)
        var e_p: Float64 = 0.0
        for k in range(BATCH * OUT):
            var d = Float64(y_buf[k]) - Float64(mu_p[k])
            e_p += 0.5 * d * d
        params_buf[i] = Scalar[dtype](Float64(saved) - h)
        _predict(x, params, mu_m_t, a_tmp_t)
        var e_m: Float64 = 0.0
        for k in range(BATCH * OUT):
            var d = Float64(y_buf[k]) - Float64(mu_m[k])
            e_m += 0.5 * d * d
        params_buf[i] = saved
        var g_fd = (e_p - e_m) / (2.0 * h)
        var g_an = Float64(grads_buf[i])
        var rel = abs(g_fd - g_an) / (abs(g_fd) + abs(g_an) + 1e-6)
        if rel > max_g_rel:
            max_g_rel = rel
    print("  [weight_grad vs FD dE/dW]      max rel error =", max_g_rel)
    var g_ok = max_g_rel < 5e-2

    print("")
    if jvp_ok and g_ok:
        print("✅ PASS — NormConvPCBlock matches finite differences")
    else:
        print("❌ FAIL")
        raise Error("NormConvPCBlock validation failed")
