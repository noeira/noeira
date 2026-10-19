"""Precision-spike schedule gate (Qi et al. 2025, arXiv:2506.23800).

Four checks, each printing COMPARED beside DIFFERING so a pass cannot be
vacuous (cf. the "0 mismatches == nothing tested" lesson):

  1. `_apply_precision_spike` scales exactly ONE level's ε slab by 1/Σ and
     leaves every other element untouched — counted both ways.
  2. The disabled default (`spike_sigma=1`) is BITWISE identical to the
     pre-change call signature. This is what keeps the other 56 pcn tests
     honest.
  3. An active spike (`spike_sigma=0.5`) actually moves the gradients —
     if this passes with 0 differing params the plumbing is dead.
  4. Energy still decreases over the inference loop with the spike on.
  5. Forward updates (Fix 2) change the INTERIOR gradients and leave the
     READOUT block bitwise alone — that split is a deliberate design choice
     (the readout has no latent above it; its ε is the loss signal).
  6. F reports the same energies as the plain path, proving the two differ
     only in the weight-gradient pass.

Run:
    pixi run mojo run -I . tests/pcn/test_precision_spike.mojo
"""

from std.math import abs as mabs
from std.memory import alloc, memset
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCReLU,
    PCTrainer,
)


comptime BATCH = 4
comptime T_INFER = 6
comptime LR_X: Float64 = 0.05

comptime NET = PCSequential[
    PCBlock[8, 6, PCIdentity],
    PCBlock[6, 6, PCReLU],
    PCBlock[6, 4, PCIdentity],
]
comptime TRAINER = PCTrainer[
    PCBlock[8, 6, PCIdentity],
    PCBlock[6, 6, PCReLU],
    PCBlock[6, 4, PCIdentity],
    dtype=dtype,
]


def main() raises:
    print("=" * 66)
    print("PCN precision-spike gate (arXiv:2506.23800 spiking schedule)")
    print("=" * 66)
    print("  arch      : 8 → 6 → 6 → 4   (N =", NET.N, "levels)")
    print("  BATCH     :", BATCH, " T_INFER:", T_INFER)

    var failures = 0

    # ── Check 1: the slab scale hits exactly one level ───────────────────────
    print("\n[1] _apply_precision_spike scales one level, nothing else")

    comptime SCRATCH = BATCH * NET.SCRATCH_OUT_DIM
    var eps_raw = alloc[Scalar[dtype]](SCRATCH).as_unsafe_any_origin()
    for i in range(SCRATCH):
        eps_raw[unsafe_offset=i] = 1.0
    var eps_t = tile[BATCH, NET.SCRATCH_OUT_DIM](eps_raw)

    comptime TARGET_LEVEL = 1
    var inv_sigma = Scalar[dtype](2.0)
    TRAINER._apply_precision_spike[BATCH](eps_t, TARGET_LEVEL, inv_sigma)

    # Expected: [BATCH*_out_offset[1], +BATCH*OUT_DIM[1]) == 2.0, rest 1.0.
    comptime LO = BATCH * NET._out_offset[TARGET_LEVEL]()
    comptime HI = LO + BATCH * NET.block_types[TARGET_LEVEL].OUT_DIM
    var n_in_slab = 0
    var n_out_slab = 0
    var bad_in = 0
    var bad_out = 0
    for i in range(SCRATCH):
        var v = Float64(eps_raw[unsafe_offset=i])
        if i >= LO and i < HI:
            n_in_slab += 1
            if mabs(v - 2.0) > 1e-9:
                bad_in += 1
        else:
            n_out_slab += 1
            if mabs(v - 1.0) > 1e-9:
                bad_out += 1
    print("      slab [", LO, ",", HI, ") of", SCRATCH, "elements")
    print("      scaled   :", n_in_slab, "compared,", bad_in, "wrong")
    print("      untouched:", n_out_slab, "compared,", bad_out, "wrong")
    if n_in_slab == 0 or n_out_slab == 0:
        print("      FAIL — vacuous: one side had nothing to compare")
        failures += 1
    elif bad_in != 0 or bad_out != 0:
        print("      FAIL")
        failures += 1
    else:
        print("      PASS")

    # A disabled call must leave the (already scaled) buffer alone.
    TRAINER._apply_precision_spike[BATCH](eps_t, TARGET_LEVEL, Scalar[dtype](1))
    TRAINER._apply_precision_spike[BATCH](eps_t, -1, Scalar[dtype](2))
    TRAINER._apply_precision_spike[BATCH](eps_t, NET.N, Scalar[dtype](2))
    var n_after = 0
    for i in range(SCRATCH):
        var v = Float64(eps_raw[unsafe_offset=i])
        var want = 2.0 if (i >= LO and i < HI) else 1.0
        if mabs(v - want) > 1e-9:
            n_after += 1
    print("      no-op guards (σ=1, idx=-1, idx=N):", n_after, "of", SCRATCH, "perturbed")
    if n_after != 0:
        print("      FAIL")
        failures += 1
    else:
        print("      PASS")

    # ── Shared setup for the driver checks ───────────────────────────────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    memset(params_buf, 0, NET.PARAM_SIZE)
    var params = tile[NET.PARAM_SIZE](params_buf)
    NET.pc_init_params[PCXavier, dtype](params)

    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_eps_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var a_below_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var z_below_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
    memset(mu_eps_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
    memset(a_below_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(z_below_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(dx_raw, 0, BATCH * NET.LATENT_DIM)

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_raw)

    var x_raw = alloc[Scalar[dtype]](BATCH * NET.IN_DIM).as_unsafe_any_origin()
    var y_raw = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    for b in range(BATCH):
        for k in range(NET.IN_DIM):
            x_raw[unsafe_offset=b * NET.IN_DIM + k] = Scalar[dtype](
                Float64(0.1 * Float64(k + 1) - 0.05 * Float64(b))
            )
        for k in range(NET.OUT_DIM):
            y_raw[unsafe_offset=b * NET.OUT_DIM + k] = (
                1.0 if k == (b % NET.OUT_DIM) else 0.0
            )
    var x_in = tile[BATCH, NET.IN_DIM](x_raw)
    var y_target = tile[BATCH, NET.OUT_DIM](y_raw)

    var g_base = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var g_off = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var g_on = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    memset(g_base, 0, NET.PARAM_SIZE)
    memset(g_off, 0, NET.PARAM_SIZE)
    memset(g_on, 0, NET.PARAM_SIZE)
    var grads_base = tile[NET.PARAM_SIZE](g_base)
    var grads_off = tile[NET.PARAM_SIZE](g_off)
    var grads_on = tile[NET.PARAM_SIZE](g_on)

    # ── Check 2: default == explicit σ=1, bitwise ────────────────────────────
    print("\n[2] disabled default is bitwise identical to spike_sigma=1")

    var r_base = TRAINER.compute_grads_only[BATCH](
        params, grads_base, latents, mu_eps_buf, a_below_buf, z_below_buf,
        dx_buf, x_in, y_target, T_INFER, Scalar[dtype](LR_X),
    )
    var r_off = TRAINER.compute_grads_only[BATCH](
        params, grads_off, latents, mu_eps_buf, a_below_buf, z_below_buf,
        dx_buf, x_in, y_target, T_INFER, Scalar[dtype](LR_X),
        Scalar[dtype](1),
    )
    var n_diff_off = 0
    for i in range(NET.PARAM_SIZE):
        if g_base[unsafe_offset=i] != g_off[unsafe_offset=i]:
            n_diff_off += 1
    print("      PARAM_SIZE:", NET.PARAM_SIZE, "compared,", n_diff_off, "differing")
    print("      energy", r_base.energy_initial, "→", r_base.energy_final)
    if n_diff_off != 0:
        print("      FAIL — the disabled path is not inert")
        failures += 1
    else:
        print("      PASS")

    # ── Check 3: an active spike moves the gradients ─────────────────────────
    print("\n[3] spike_sigma=0.5 changes the gradients (non-vacuity)")

    var r_on = TRAINER.compute_grads_only[BATCH](
        params, grads_on, latents, mu_eps_buf, a_below_buf, z_below_buf,
        dx_buf, x_in, y_target, T_INFER, Scalar[dtype](LR_X),
        Scalar[dtype](0.5),
    )
    var n_diff_on = 0
    var max_rel = Float64(0.0)
    for i in range(NET.PARAM_SIZE):
        var a = Float64(g_base[unsafe_offset=i])
        var b = Float64(g_on[unsafe_offset=i])
        if a != b:
            n_diff_on += 1
        var den = mabs(a) + mabs(b) + 1e-12
        var rel = mabs(a - b) / den
        if rel > max_rel:
            max_rel = rel
    print("      PARAM_SIZE:", NET.PARAM_SIZE, "compared,", n_diff_on, "differing")
    print("      max normalized delta:", max_rel)
    if n_diff_on == 0:
        print("      FAIL — spike is a no-op; the schedule never fired")
        failures += 1
    else:
        print("      PASS")

    # ── Check 4: energy still descends with the spike on ─────────────────────
    print("\n[4] energy descends with the spike on")
    print("      energy", r_on.energy_initial, "→", r_on.energy_final)
    if not (r_on.energy_final < r_on.energy_initial):
        print("      FAIL — inference did not reduce energy")
        failures += 1
    else:
        print("      PASS")

    # ── Check 5/6: forward updates (Fix 2) ───────────────────────────────────
    print("\n[5] forward updates move interior grads, not the readout block")

    var lat0_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    memset(lat0_raw, 0, BATCH * NET.LATENT_DIM)
    var latents_0 = tile[BATCH, NET.LATENT_DIM](lat0_raw)

    var g_fwd = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    memset(g_fwd, 0, NET.PARAM_SIZE)
    var grads_fwd = tile[NET.PARAM_SIZE](g_fwd)

    var r_fwd = TRAINER.compute_grads_only_fwd[BATCH](
        params, grads_fwd, latents, latents_0, mu_eps_buf, a_below_buf,
        z_below_buf, dx_buf, x_in, y_target, T_INFER, Scalar[dtype](LR_X),
    )

    comptime RO = NET.N - 1
    comptime RO_LO = NET._param_offset[RO]()
    comptime RO_HI = RO_LO + NET.block_types[RO].PARAM_SIZE
    var n_ro = 0
    var bad_ro = 0
    var n_int = 0
    var moved_int = 0
    for i in range(NET.PARAM_SIZE):
        if i >= RO_LO and i < RO_HI:
            n_ro += 1
            if g_base[unsafe_offset=i] != g_fwd[unsafe_offset=i]:
                bad_ro += 1
        else:
            n_int += 1
            if g_base[unsafe_offset=i] != g_fwd[unsafe_offset=i]:
                moved_int += 1
    print("      readout block [", RO_LO, ",", RO_HI, "):", n_ro,
          "compared,", bad_ro, "changed (want 0)")
    print("      interior params            :", n_int,
          "compared,", moved_int, "changed (want > 0)")
    if n_ro == 0 or n_int == 0:
        print("      FAIL — vacuous: one side had nothing to compare")
        failures += 1
    elif bad_ro != 0:
        print("      FAIL — forward update touched the readout block")
        failures += 1
    elif moved_int == 0:
        print("      FAIL — forward update changed nothing")
        failures += 1
    else:
        print("      PASS")

    print("\n[6] F reports the same energies as the plain path")
    print("      plain:", r_base.energy_initial, "→", r_base.energy_final)
    print("      fwd  :", r_fwd.energy_initial, "→", r_fwd.energy_final)
    if (r_fwd.energy_initial != r_base.energy_initial
            or r_fwd.energy_final != r_base.energy_final):
        print("      FAIL — the two paths diverged before the grad pass")
        failures += 1
    else:
        print("      PASS")

    print("\n" + "=" * 66)
    if failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print("FAILURES:", failures)
        raise Error("precision-spike gate failed")
