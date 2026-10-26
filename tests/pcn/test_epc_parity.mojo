"""ePC gate — the equilibrium-equivalence test.

`compute_grads_only_epc` reparameterizes the inference from the STATES to the
ERRORS (arXiv:2505.20137; derivation in docs/PCN_EPC_DERIVATION.md). The
paper's load-bearing claim is that ePC computes EXACT PC weight gradients — the
same ones sPC reaches — just in far fewer iterations. Since ε = x − μ(x_below)
is a bijective reparameterization at fixed input, the two have the SAME
stationary points, so at equilibrium the weight gradients must agree.

That is the gate. It is also the test that catches a sign error, which is the
live risk here: our `eps_compute` writes ε = x_above − μ, so the readout ε is
`y − y_pred` = MINUS the loss gradient, and the whole top-down sweep hangs off
absorbing that correctly.

Checks:
  1. ε = 0 reconstructs exactly the `init_latents` forward sweep (bitwise).
  2. EQUILIBRIUM: sPC and ePC, both run to convergence from identical params,
     produce the same weight gradients.
  3. NON-VACUITY: at the small budget both methods actually run at (T=5),
     they DISAGREE — otherwise check 2 would be trivially satisfiable.
  4. ePC descends its energy.

Run:
    pixi run mojo run -I . tests/pcn/test_epc_parity.mojo
"""

from std.math import abs as mabs, sqrt
from std.memory import alloc, unsafe_memset
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn import (
    PCBlock, PCSequential, PCIdentity, PCReLU, PCTrainer,
)

comptime BATCH = 4
comptime T_EQ = 4000            # to equilibrium
comptime T_SMALL = 5            # the regime the reference actually runs
comptime LR_EQ: Float64 = 0.02

comptime NET = PCSequential[
    PCBlock[6, 8, PCIdentity],
    PCBlock[8, 5, PCReLU],
    PCBlock[5, 3, PCIdentity],
]
comptime TRAINER = PCTrainer[
    PCBlock[6, 8, PCIdentity],
    PCBlock[8, 5, PCReLU],
    PCBlock[5, 3, PCIdentity],
    dtype=dtype,
]


def main() raises:
    print("=" * 68)
    print("ePC gate — equilibrium equivalence with sPC")
    print("=" * 68)
    print("  arch 6→8→5→3   BATCH", BATCH, "  PARAM_SIZE", NET.PARAM_SIZE)

    var failures = 0

    var params_raw = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    unsafe_memset(params_raw, 0, NET.PARAM_SIZE)
    var params = tile[NET.PARAM_SIZE](params_raw)
    NET.pc_init_params[PCXavier, dtype](params)

    var g_s = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var g_e = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var g_s5 = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var g_e5 = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    for b in [g_s, g_e, g_s5, g_e5]:
        unsafe_memset(b, 0, NET.PARAM_SIZE)
    var grads_s = tile[NET.PARAM_SIZE](g_s)
    var grads_e = tile[NET.PARAM_SIZE](g_e)
    var grads_s5 = tile[NET.PARAM_SIZE](g_s5)
    var grads_e5 = tile[NET.PARAM_SIZE](g_e5)

    var lat_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var ref_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var err_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var dx_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var ab_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var zb_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    unsafe_memset(lat_raw, 0, BATCH * NET.LATENT_DIM)
    unsafe_memset(ref_raw, 0, BATCH * NET.LATENT_DIM)
    unsafe_memset(err_raw, 0, BATCH * NET.LATENT_DIM)
    unsafe_memset(dx_raw, 0, BATCH * NET.LATENT_DIM)
    unsafe_memset(mu_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
    unsafe_memset(ab_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(zb_raw, 0, BATCH * NET.SCRATCH_IN_DIM)

    var latents = tile[BATCH, NET.LATENT_DIM](lat_raw)
    var ref_lat = tile[BATCH, NET.LATENT_DIM](ref_raw)
    var errors = tile[BATCH, NET.LATENT_DIM](err_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_raw)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](ab_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](zb_raw)

    var x_raw = alloc[Scalar[dtype]](BATCH * NET.IN_DIM).as_unsafe_any_origin()
    var y_raw = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    for b in range(BATCH):
        for k in range(NET.IN_DIM):
            x_raw[unsafe_offset=b * NET.IN_DIM + k] = Scalar[dtype](
                0.2 * Float64(k + 1) - 0.1 * Float64(b)
            )
        for k in range(NET.OUT_DIM):
            y_raw[unsafe_offset=b * NET.OUT_DIM + k] = 1.0 if k == (b % NET.OUT_DIM) else 0.0
    var x_in = tile[BATCH, NET.IN_DIM](x_raw)
    var y_target = tile[BATCH, NET.OUT_DIM](y_raw)

    # ── 1. eps = 0 reconstructs the forward sweep ────────────────────────────
    print("\n[1] ε=0 reconstruction == init_latents forward sweep")
    NET.init_latents[BATCH, dtype](x_in, params, ref_lat)
    for i in range(BATCH * NET.LATENT_DIM):
        err_raw[unsafe_offset=i] = 0
    TRAINER._epc_reconstruct[BATCH](
        x_in, y_target, params, errors, latents, mu_eps_buf, a_below_buf
    )
    var n_lat = 0
    var n_bad = 0
    for i in range(BATCH * NET.LATENT_DIM):
        n_lat += 1
        if lat_raw[unsafe_offset=i] != ref_raw[unsafe_offset=i]:
            n_bad += 1
    print("      LATENT_DIM*BATCH:", n_lat, "compared,", n_bad, "differing")
    if n_lat == 0 or n_bad != 0:
        print("      FAIL"); failures += 1
    else:
        print("      PASS")

    # ── 2. equilibrium equivalence ───────────────────────────────────────────
    print("\n[2] sPC(T=", T_EQ, ") vs ePC(T=", T_EQ, ") weight gradients")
    var r_s = TRAINER.compute_grads_only[BATCH](
        params, grads_s, latents, mu_eps_buf, a_below_buf, z_below_buf,
        dx_buf, x_in, y_target, T_EQ, Scalar[dtype](LR_EQ),
    )
    var r_e = TRAINER.compute_grads_only_epc[BATCH](
        params, grads_e, errors, latents, mu_eps_buf, a_below_buf,
        z_below_buf, dx_buf, x_in, y_target, T_EQ, Scalar[dtype](LR_EQ),
    )
    print("      sPC energy", r_s.energy_initial, "→", r_s.energy_final)
    print("      ePC energy", r_e.energy_initial, "→", r_e.energy_final)

    var max_abs: Float64 = 0.0
    var sum_s: Float64 = 0.0
    var sum_e: Float64 = 0.0
    for i in range(NET.PARAM_SIZE):
        var a = Float64(g_s[unsafe_offset=i])
        var bb = Float64(g_e[unsafe_offset=i])
        var d = mabs(a - bb)
        if d > max_abs:
            max_abs = d
        sum_s += a * a
        sum_e += bb * bb
    var nrm = sqrt(sum_s) + sqrt(sum_e) + 1e-30
    var rel = max_abs / (nrm / Float64(NET.PARAM_SIZE) + 1e-30)
    print("      max |g_sPC − g_ePC| :", max_abs)
    print("      ‖g_sPC‖ =", sqrt(sum_s), "  ‖g_ePC‖ =", sqrt(sum_e))
    print("      normalized          :", rel)
    if sqrt(sum_s) < 1e-12 or sqrt(sum_e) < 1e-12:
        print("      FAIL — vacuous: a gradient vector is ~zero"); failures += 1
    elif rel > 1.0e-2:
        print("      FAIL — the two do NOT agree at equilibrium"); failures += 1
    else:
        print("      PASS")

    # ── 3. non-vacuity at the small budget ───────────────────────────────────
    print("\n[3] sPC(T=", T_SMALL, ") vs ePC(T=", T_SMALL, ") must DIFFER")
    var rs5 = TRAINER.compute_grads_only[BATCH](
        params, grads_s5, latents, mu_eps_buf, a_below_buf, z_below_buf,
        dx_buf, x_in, y_target, T_SMALL, Scalar[dtype](LR_EQ),
    )
    var re5 = TRAINER.compute_grads_only_epc[BATCH](
        params, grads_e5, errors, latents, mu_eps_buf, a_below_buf,
        z_below_buf, dx_buf, x_in, y_target, T_SMALL, Scalar[dtype](LR_EQ),
    )
    var n_diff = 0
    for i in range(NET.PARAM_SIZE):
        if g_s5[unsafe_offset=i] != g_e5[unsafe_offset=i]:
            n_diff += 1
    print("      PARAM_SIZE:", NET.PARAM_SIZE, "compared,", n_diff, "differing")
    print("      sPC readout loss", rs5.output_loss_final,
          " ePC readout loss", re5.output_loss_final)
    if n_diff == 0:
        print("      FAIL — identical at T=5; check 2 would be vacuous"); failures += 1
    else:
        print("      PASS")

    # ── 4. ePC descends ──────────────────────────────────────────────────────
    print("\n[4] ePC energy descends")
    print("      ", re5.energy_initial, "→", re5.energy_final)
    if not (re5.energy_final < re5.energy_initial):
        print("      FAIL"); failures += 1
    else:
        print("      PASS")

    print("\n" + "=" * 68)
    if failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print("FAILURES:", failures)
        raise Error("ePC gate failed")
