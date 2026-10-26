"""ConvPCBlock parity spike (P0) — see docs/PCN_CONV_DESIGN.md.

Validates the convolutional PC math mapping by checking ConvPCBlock's three
core ops against an INDEPENDENT naive direct-convolution oracle (two
implementations agreeing). Per Salvatori et al. 2021, PC inference on a conv
graph reproduces backprop gradients exactly, so the conv backward IS an exact
oracle (modulo the PC sign convention and the activation-first ordering):

  predict   μ          == conv_eval(a_below)              (a_below = ACT(x_below))
  pull_back z_below    == conv_vjp grad_input             (w.r.t a_below)
  weight_grad W-part   == −(conv_vjp grad_W)              (PC bakes the −sign)
  weight_grad b-part   == −(conv_vjp grad_b)

The oracle is a self-contained naive nested-loop conv (eval + grad_input +
grad_W + grad_b) — a different implementation from ConvPCBlock's im2col + BLAS
path, so the agreement is a genuine cross-check. (It replaced the legacy
`nn.autodiff.primitives.conv2d.Conv2D` oracle during the nn re-architecture;
PCN is now free of legacy `nn`.)

Layout conventions (match ConvPCBlock):
  a/x:    [B, IC*H*W],  index b*IN + ic*H*W + ih*W + iw
  μ/eps:  [B, OC*oh*ow], index b*OUT + oc*oh*ow + ohh*ow + oww
  W:      params[oc*col + ic*K*K + kh*K + kw],  col = IC*K*K
  bias:   params[OC*col + oc]

Run:
    pixi run mojo run -I . tests/pcn/test_conv_pc_block_parity.mojo
"""

from std.memory import alloc
from std.math import sin
from noeira.core.tiles import tile

from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_conv_block import ConvPCBlock
from noeira.experimental.pcn import PCReLU

comptime dtype = DType.float32
comptime TOL: Float32 = 1e-4


def _max_abs_diff(
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    b: Pointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
) -> Float32:
    var m: Float32 = 0.0
    for i in range(n):
        var d = a[i] - b[i]
        var ad = d if d >= 0 else -d
        if ad > m:
            m = ad
    return m


# ── Naive direct-conv oracle (independent of ConvPCBlock's im2col+BLAS) ──────


def _conv_eval[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, BATCH: Int
](
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    params: Pointer[Scalar[dtype], MutAnyOrigin],
    mu: Pointer[Scalar[dtype], MutAnyOrigin],
):
    comptime OH = (H + 2 * P - K) // S + 1
    comptime OW = (W + 2 * P - K) // S + 1
    comptime COL = IC * K * K
    comptime IN = IC * H * W
    comptime OUT = OC * OH * OW
    comptime WSZ = OC * COL
    for b in range(BATCH):
        for oc in range(OC):
            for oh in range(OH):
                for ow in range(OW):
                    var acc = Float64(params[WSZ + oc])  # bias
                    for ic in range(IC):
                        for kh in range(K):
                            for kw in range(K):
                                var ih = oh * S - P + kh
                                var iw = ow * S - P + kw
                                if 0 <= ih < H and 0 <= iw < W:
                                    var wv = params[
                                        oc * COL + ic * K * K + kh * K + kw
                                    ]
                                    var av = a[
                                        b * IN + ic * H * W + ih * W + iw
                                    ]
                                    acc += Float64(wv) * Float64(av)
                    mu[b * OUT + oc * OH * OW + oh * OW + ow] = Scalar[dtype](
                        acc
                    )


def _conv_vjp[
    IC: Int, OC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, BATCH: Int
](
    eps: Pointer[Scalar[dtype], MutAnyOrigin],
    a: Pointer[Scalar[dtype], MutAnyOrigin],
    params: Pointer[Scalar[dtype], MutAnyOrigin],
    gi: Pointer[Scalar[dtype], MutAnyOrigin],
    gp: Pointer[Scalar[dtype], MutAnyOrigin],
):
    """Standard conv backward → grad_input (gi), grad_W + grad_b (gp slab)."""
    comptime OH = (H + 2 * P - K) // S + 1
    comptime OW = (W + 2 * P - K) // S + 1
    comptime COL = IC * K * K
    comptime IN = IC * H * W
    comptime OUT = OC * OH * OW
    comptime WSZ = OC * COL
    for i in range(BATCH * IN):
        gi[i] = Scalar[dtype](0)
    for i in range(WSZ + OC):
        gp[i] = Scalar[dtype](0)
    for b in range(BATCH):
        for oc in range(OC):
            for oh in range(OH):
                for ow in range(OW):
                    var g = eps[b * OUT + oc * OH * OW + oh * OW + ow]
                    gp[WSZ + oc] = gp[WSZ + oc] + g  # grad_b
                    for ic in range(IC):
                        for kh in range(K):
                            for kw in range(K):
                                var ih = oh * S - P + kh
                                var iw = ow * S - P + kw
                                if 0 <= ih < H and 0 <= iw < W:
                                    var wi = oc * COL + ic * K * K + kh * K + kw
                                    var ai = b * IN + ic * H * W + ih * W + iw
                                    gp[wi] = gp[wi] + g * a[ai]  # grad_W
                                    gi[ai] = gi[ai] + g * params[wi]  # grad_in


def run_parity[
    IC: Int,
    OC: Int,
    K: Int,
    S: Int,
    P: Int,
    H: Int,
    W: Int,
    BATCH: Int,
](label: String) raises -> Bool:
    comptime CB = ConvPCBlock[IC, OC, K, S, P, H, W, PCReLU]

    comptime IN = CB.IN_DIM
    comptime OUT = CB.OUT_DIM
    comptime PSZ = CB.PARAM_SIZE
    comptime W_SIZE = OC * CB.col_size

    # ── Buffers ──────────────────────────────────────────────────────────────
    var x_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var a_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var mu_buf = alloc[Scalar[dtype]](BATCH * OUT).as_unsafe_any_origin()
    var eps_buf = alloc[Scalar[dtype]](BATCH * OUT).as_unsafe_any_origin()
    var params_buf = alloc[Scalar[dtype]](PSZ).as_unsafe_any_origin()
    var z_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](PSZ).as_unsafe_any_origin()

    var mu_oracle_buf = alloc[Scalar[dtype]](BATCH * OUT).as_unsafe_any_origin()
    var gi_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var gp_buf = alloc[Scalar[dtype]](PSZ).as_unsafe_any_origin()

    var x = tile[BATCH, IN](x_buf)
    var a = tile[BATCH, IN](a_buf)
    var mu = tile[BATCH, OUT](mu_buf)
    var eps = tile[BATCH, OUT](eps_buf)
    var params = tile[PSZ](params_buf)
    var z = tile[BATCH, IN](z_buf)
    var grads = tile[PSZ](grads_buf)

    # ── Deterministic inputs (varied sign so ReLU is exercised) ───────────────
    for i in range(BATCH * IN):
        x_buf[i] = Scalar[dtype](sin(Float32(i) * 0.7 + 0.3) * 1.5)
    for i in range(BATCH * OUT):
        eps_buf[i] = Scalar[dtype](sin(Float32(i) * 1.1 + 1.7))
    CB.pc_init_params[PCXavier, dtype](params)

    # ── ConvPCBlock ops ───────────────────────────────────────────────────────
    CB.predict[BATCH, dtype](x, params, mu, a)
    CB.pull_back[BATCH, dtype](eps, params, z)
    CB.weight_grad[BATCH, dtype](eps, a, grads)

    # ── Oracle: naive conv eval(a) + naive conv vjp(grad_output=eps) ──────────
    _conv_eval[IC, OC, K, S, P, H, W, BATCH](a_buf, params_buf, mu_oracle_buf)
    _conv_vjp[IC, OC, K, S, P, H, W, BATCH](
        eps_buf, a_buf, params_buf, gi_buf, gp_buf
    )

    # ── Compare ───────────────────────────────────────────────────────────────
    var d_predict = _max_abs_diff(mu_buf, mu_oracle_buf, BATCH * OUT)
    var d_pullback = _max_abs_diff(z_buf, gi_buf, BATCH * IN)

    # weight_grad parity: ConvPCBlock grad == −(oracle grad). Negate oracle.
    var neg_gp = alloc[Scalar[dtype]](PSZ).as_unsafe_any_origin()
    for i in range(PSZ):
        neg_gp[i] = -gp_buf[i]
    var d_wgrad_W = _max_abs_diff(grads_buf, neg_gp, W_SIZE)
    var d_wgrad_b = _max_abs_diff(grads_buf + W_SIZE, neg_gp + W_SIZE, OC)
    neg_gp.free()

    print("── " + label + " ──")
    print("  IN=" + String(IN) + " OUT=" + String(OUT) + " out_h=" + String(
        CB.out_h
    ) + " out_w=" + String(CB.out_w))
    print("  predict   max|Δ| =", d_predict)
    print("  pull_back max|Δ| =", d_pullback)
    print("  weight_gW max|Δ| =", d_wgrad_W)
    print("  weight_gb max|Δ| =", d_wgrad_b)

    var ok = (
        d_predict < TOL
        and d_pullback < TOL
        and d_wgrad_W < TOL
        and d_wgrad_b < TOL
    )

    x_buf.free()
    a_buf.free()
    mu_buf.free()
    eps_buf.free()
    params_buf.free()
    z_buf.free()
    grads_buf.free()
    mu_oracle_buf.free()
    gi_buf.free()
    gp_buf.free()
    return ok


def main() raises:
    print("ConvPCBlock ↔ naive-conv oracle parity spike\n")
    var all_ok = True

    # Config A: same-size (stride 1, pad 1) — boundary padding exercised.
    all_ok = run_parity[2, 3, 3, 1, 1, 4, 4, 2]("A  stride=1 pad=1 4x4") and all_ok

    # Config B: downsampling (stride 2, pad 1) — strided.
    all_ok = run_parity[2, 3, 3, 2, 1, 5, 5, 2]("B  stride=2 pad=1 5x5") and all_ok

    # Config C: no pad, 1 channel→multi, batch 3.
    all_ok = run_parity[1, 4, 3, 1, 0, 6, 6, 3]("C  stride=1 pad=0 6x6") and all_ok

    print("")
    if all_ok:
        print("✅ PASS — ConvPCBlock matches naive-conv oracle within", TOL)
    else:
        print("❌ FAIL — parity mismatch")
        raise Error("ConvPCBlock parity failed")
