"""Backprop-control gate — `compute_grads_only_bp` against finite differences.

`PCTrainer.compute_grads_only_bp` is the control arm for every PC accuracy
number: the same network, kernels and weight pass, with the relaxation
replaced by one top-down sweep. It claims to return the EXACT gradient of the
readout loss L = 0.5·Σ‖y − ŷ‖² of the feedforward function `forward_eval`.
That claim is what this file checks, on a conv + conv + linear stack (the
shape of the CIFAR-10 net) in float64.

Checks:
  1. FINITE DIFFERENCES: central differences of L through `forward_eval`, over
     EVERY parameter, agree with the returned gradient.
  2. NON-VACUITY: every block has a non-zero gradient, so check 1 is not
     passing on zeros in the lower levels (the failure a broken top-down
     sweep would produce).
  3. IT IS NOT THE PC GRADIENT: sPC at T=6 from the same parameters differs,
     so the two arms of a PC-vs-BP comparison really are two arms.

Run:
    pixi run mojo run -I . tests/pcn/test_bp_grad_parity.mojo
"""

from std.math import abs as mabs, sqrt
from std.memory import alloc, unsafe_memset
from noeira.core.tiles import Tile, tile

from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn import (
    PCBlock, PCSequential, PCIdentity, PCReLU, PCTrainer,
)
from noeira.experimental.pcn.pc_conv_block import ConvPCBlock

comptime dtype = DType.float64
comptime BATCH = 3
comptime FD_H: Float64 = 1.0e-6
comptime T_PC = 6
comptime LR_X: Float64 = 0.025

# 2x8x8 -> 3x4x4 -> 4x2x2 -> 3 : conv, conv, linear readout.
comptime NET = PCSequential[
    ConvPCBlock[2, 3, 3, 2, 1, 8, 8, PCIdentity],
    ConvPCBlock[3, 4, 3, 2, 1, 4, 4, PCReLU],
    PCBlock[16, 3, PCReLU],
]
comptime TRAINER = PCTrainer[
    ConvPCBlock[2, 3, 3, 2, 1, 8, 8, PCIdentity],
    ConvPCBlock[3, 4, 3, 2, 1, 4, 4, PCReLU],
    PCBlock[16, 3, PCReLU],
    dtype=dtype,
]


def _loss(
    x_in: Tile[dtype, BATCH, NET.IN_DIM],
    y_raw: Pointer[Scalar[dtype], MutAnyOrigin],
    params: Tile[dtype, NET.PARAM_SIZE],
    y_hat: Tile[dtype, BATCH, NET.OUT_DIM],
) -> Float64:
    """L = 0.5·Σ_b Σ_k (y − ŷ)² through the deployed feedforward path."""
    NET.forward_eval[BATCH, dtype](x_in, params, y_hat)
    var total: Float64 = 0.0
    for i in range(BATCH * NET.OUT_DIM):
        var d = Float64(y_raw[unsafe_offset=i]) - Float64(y_hat.ptr[unsafe_offset=i])
        total += d * d
    return 0.5 * total


def main() raises:
    print("=" * 68)
    print("Backprop-control gate — compute_grads_only_bp vs finite differences")
    print("=" * 68)
    print("  arch conv(2→3) conv(3→4) linear(16→3)   BATCH", BATCH,
          "  PARAM_SIZE", NET.PARAM_SIZE)

    var failures = 0

    var params_raw = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    unsafe_memset(params_raw, 0, NET.PARAM_SIZE)
    var params = tile[NET.PARAM_SIZE](params_raw)
    NET.pc_init_params[PCXavier, dtype](params)
    # Biases start at zero; move them so their gradients are exercised too.
    for i in range(NET.PARAM_SIZE):
        if params_raw[unsafe_offset=i] == 0:
            params_raw[unsafe_offset=i] = Scalar[dtype](
                0.05 * Float64((i * 7) % 5 - 2)
            )

    var g_bp = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var g_pc = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    unsafe_memset(g_bp, 0, NET.PARAM_SIZE)
    unsafe_memset(g_pc, 0, NET.PARAM_SIZE)
    var grads_bp = tile[NET.PARAM_SIZE](g_bp)
    var grads_pc = tile[NET.PARAM_SIZE](g_pc)

    var lat_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var dx_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var ab_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var zb_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var out_raw = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    unsafe_memset(lat_raw, 0, BATCH * NET.LATENT_DIM)
    unsafe_memset(dx_raw, 0, BATCH * NET.LATENT_DIM)
    unsafe_memset(mu_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
    unsafe_memset(ab_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(zb_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(out_raw, 0, BATCH * NET.OUT_DIM)
    var latents = tile[BATCH, NET.LATENT_DIM](lat_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_raw)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](ab_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](zb_raw)
    var y_hat = tile[BATCH, NET.OUT_DIM](out_raw)

    var x_raw = alloc[Scalar[dtype]](BATCH * NET.IN_DIM).as_unsafe_any_origin()
    var y_raw = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    for b in range(BATCH):
        for k in range(NET.IN_DIM):
            x_raw[unsafe_offset=b * NET.IN_DIM + k] = Scalar[dtype](
                0.3 * Float64((k * 5 + b * 3) % 7) - 0.8
            )
        for k in range(NET.OUT_DIM):
            y_raw[unsafe_offset=b * NET.OUT_DIM + k] = 1.0 if k == (b % NET.OUT_DIM) else 0.0
    var x_in = tile[BATCH, NET.IN_DIM](x_raw)
    var y_target = tile[BATCH, NET.OUT_DIM](y_raw)

    var r_bp = TRAINER.compute_grads_only_bp[BATCH](
        params, grads_bp, latents, mu_eps_buf, a_below_buf, z_below_buf,
        x_in, y_target,
    )
    var l0 = _loss(x_in, y_raw, params, y_hat)
    print("\n  loss via forward_eval :", l0)
    print("  loss the trainer saw  :", r_bp.output_loss_final)

    # ── 1. finite differences over every parameter ──────────────────────────
    print("\n[1] central differences, h =", FD_H, ", all", NET.PARAM_SIZE, "parameters")
    var max_abs: Float64 = 0.0
    var sum_fd: Float64 = 0.0
    var sum_bp: Float64 = 0.0
    var dot: Float64 = 0.0
    for i in range(NET.PARAM_SIZE):
        var p0 = params_raw[unsafe_offset=i]
        params_raw[unsafe_offset=i] = p0 + Scalar[dtype](FD_H)
        var lp = _loss(x_in, y_raw, params, y_hat)
        params_raw[unsafe_offset=i] = p0 - Scalar[dtype](FD_H)
        var lm = _loss(x_in, y_raw, params, y_hat)
        params_raw[unsafe_offset=i] = p0
        var fd = (lp - lm) / (2.0 * FD_H)
        var bp = Float64(g_bp[unsafe_offset=i])
        var d = mabs(fd - bp)
        if d > max_abs:
            max_abs = d
        sum_fd += fd * fd
        sum_bp += bp * bp
        dot += fd * bp
    var cosine = dot / (sqrt(sum_fd) * sqrt(sum_bp) + 1e-300)
    var rel = max_abs / (sqrt(sum_fd) + 1e-300)
    print("      ‖g_fd‖ =", sqrt(sum_fd), "  ‖g_bp‖ =", sqrt(sum_bp))
    print("      cosine              :", cosine)
    print("      max |g_fd − g_bp|   :", max_abs, "  (relative to ‖g_fd‖:", rel, ")")
    if mabs(l0 - r_bp.output_loss_final) > 1e-9 * (1.0 + l0):
        print("      FAIL — the trainer's loss is not forward_eval's loss"); failures += 1
    elif sqrt(sum_fd) < 1e-6:
        print("      FAIL — vacuous: the finite-difference gradient is ~zero"); failures += 1
    elif rel > 1.0e-6 or cosine < 1.0 - 1.0e-10:
        print("      FAIL — not the gradient of the readout loss"); failures += 1
    else:
        print("      PASS")

    # ── 2. every block carries gradient ─────────────────────────────────────
    print("\n[2] per-block gradient norm (a broken sweep zeroes the lower ones)")
    comptime for i in range(NET.N):
        var s: Float64 = 0.0
        for j in range(NET.block_types[i].PARAM_SIZE):
            var v = Float64(g_bp[unsafe_offset=NET._param_offset[i]() + j])
            s += v * v
        print("      block", i, " ‖g‖ =", sqrt(s))
        if sqrt(s) < 1e-8:
            print("      FAIL — block", i, "has no gradient"); failures += 1

    # ── 3. it is not the PC gradient at the budget we train with ────────────
    print("\n[3] BP vs sPC(T=", T_PC, ") must DIFFER")
    var r_pc = TRAINER.compute_grads_only[BATCH](
        params, grads_pc, latents, mu_eps_buf, a_below_buf, z_below_buf,
        dx_buf, x_in, y_target, T_PC, Scalar[dtype](LR_X),
    )
    var sum_pc: Float64 = 0.0
    var dot_pc: Float64 = 0.0
    for i in range(NET.PARAM_SIZE):
        var a = Float64(g_pc[unsafe_offset=i])
        sum_pc += a * a
        dot_pc += a * Float64(g_bp[unsafe_offset=i])
    var cos_pc = dot_pc / (sqrt(sum_pc) * sqrt(sum_bp) + 1e-300)
    print("      ‖g_sPC‖ =", sqrt(sum_pc), "  cosine(BP, sPC) =", cos_pc)
    print("      sPC readout loss after relaxation:", r_pc.output_loss_final)
    if cos_pc > 1.0 - 1.0e-6:
        print("      FAIL — indistinguishable from sPC; the comparison would be vacuous")
        failures += 1
    else:
        print("      PASS")

    print("\n" + "=" * 68)
    if failures == 0:
        print("ALL CHECKS PASSED")
    else:
        print("FAILURES:", failures)
        raise Error("backprop-control gate failed")
