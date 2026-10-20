"""Relaxation stability sweep — does a smaller lr_x settle the bottom levels?

P14 found a non-monotone per-level error profile on BOTH conv nets: a clean
geometric top-down decay over the upper levels, a 3-4 order-of-magnitude
collapse in the middle, and a bottom pool 1e4x larger than the level above it
whose eps moves non-monotonically across t rather than settling. The reading
was that the x-update is UNSTABLE at the high-dimensional early conv levels —
a step-size property, not a credit-assignment one.

This is the test of that reading. ONE set of weights (trained at the P10
recipe's own lr_x), one batch, and the relaxation re-run across
lr_x x T_infer. Only the inference loop varies.

  lr_x in {0.025 (P10 recipe), 0.005, 0.001}  x  T in {20 (recipe), 100}

VACUITY GUARD — this is the trap in this experiment. A smaller lr_x moves the
latents less, so eps at the bottom shrinks for the trivial reason that nothing
happened. "Bottom pool gone" is only meaningful if the relaxation ALSO did its
job. So every arm reports the total energy at t=0 and t=T and the readout eps:
an arm whose energy barely descends has not fixed anything, it has just idled.
That is why T=100 is swept alongside — a small lr_x needs more steps to reach
the same place.

Read:
  * bottom pool gone AND energy descends as far as the baseline
        => the relaxation was unstable; lr_x is the wall, and the whole
           conv-PCN line has been training on an unconverged inference loop.
  * bottom pool persists at every lr_x
        => not a step-size problem; P14's reading is wrong and the bottom
           error is structural.
  * bottom pool gone but energy stalls
        => vacuous arm, no conclusion.

Run:
    pixi run mojo run -I . tests/pcn/test_pc_relaxation_lrx_sweep.mojo
"""

from std.math import sqrt
from std.memory import alloc, unsafe_memset
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.nn.datasets.cifar10 import CIFAR10
from noeira.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCReLU,
    PCTrainer,
)
from noeira.experimental.pcn.pc_conv_block import ConvPCBlock
from noeira.core.fmt import fit

comptime BATCH = 125
comptime T_INFER = 20
comptime LR_X: Float32 = 0.025
comptime ADAM_LR: Float64 = 0.0002
comptime WARM_STEPS = 80          # one epoch of the 10k screen
comptime IN = 3 * 32 * 32

comptime NET = PCSequential[
    ConvPCBlock[3, 32, 3, 2, 1, 32, 32, PCIdentity],
    ConvPCBlock[32, 64, 3, 2, 1, 16, 16, PCReLU],
    ConvPCBlock[64, 128, 3, 2, 1, 8, 8, PCReLU],
    ConvPCBlock[128, 128, 3, 1, 1, 4, 4, PCReLU],
    PCBlock[2048, 512, PCReLU],
    PCBlock[512, 10, PCIdentity],
]
comptime TRAINER = PCTrainer[
    ConvPCBlock[3, 32, 3, 2, 1, 32, 32, PCIdentity],
    ConvPCBlock[32, 64, 3, 2, 1, 16, 16, PCReLU],
    ConvPCBlock[64, 128, 3, 2, 1, 8, 8, PCReLU],
    ConvPCBlock[128, 128, 3, 1, 1, 4, 4, PCReLU],
    PCBlock[2048, 512, PCReLU],
    PCBlock[512, 10, PCIdentity],
    dtype=dtype,
]
comptime OPT = PCAdam[LR=ADAM_LR]


def _print_profile(
    tag: String,
    eps_raw: Pointer[Scalar[dtype], MutAnyOrigin],
):
    """rms(ε_l) per level + the adjacent-level ratio r_l = rms_{l+1}/rms_l."""
    var rms = List[Float64]()
    comptime for i in range(NET.N):
        comptime LO = BATCH * NET._out_offset[i]()
        comptime CNT = BATCH * NET.block_types[i].OUT_DIM
        var acc: Float64 = 0.0
        for k in range(CNT):
            var v = Float64(eps_raw[LO + k])
            acc += v * v
        rms.append(sqrt(acc / Float64(CNT)))

    var line = String("    ") + fit(tag, 10) + " rms:"
    for i in range(len(rms)):
        line += " " + fit(String(rms[i]), 9)
    print(line)

    var rline = String("    ") + fit(String(""), 10) + " r_l:"
    for i in range(len(rms) - 1):
        # r > 1 means the level ABOVE carries more error than this one,
        # i.e. error shrinks on the way down. Constant r = exponential.
        var den = rms[i] + 1e-30
        rline += " " + fit(String(rms[i + 1] / den), 9)
    print(rline)


def _probe(
    tag: String,
    lr_x: Float64,
    t_infer: Int,
    eps_raw: Pointer[Scalar[dtype], MutAnyOrigin],
    x_batch: Tile[dtype, BATCH, NET.IN_DIM],
    y_batch: Tile[dtype, BATCH, NET.OUT_DIM],
    params: Tile[dtype, NET.PARAM_SIZE],
    latents: Tile[dtype, BATCH, NET.LATENT_DIM],
    mu_eps_buf: Tile[dtype, BATCH, NET.SCRATCH_OUT_DIM],
    a_below_buf: Tile[dtype, BATCH, NET.SCRATCH_IN_DIM],
    z_below_buf: Tile[dtype, BATCH, NET.SCRATCH_IN_DIM],
    dx_buf: Tile[dtype, BATCH, NET.LATENT_DIM],
) raises:
    """Relax at (lr_x, t_infer) from a fresh forward sweep; report descent."""
    NET.init_latents[BATCH, dtype](x_batch, params, latents)
    TRAINER._forward_eps[BATCH](
        x_batch, y_batch, params, latents, mu_eps_buf, a_below_buf
    )
    var e0 = TRAINER._total_energy[BATCH](mu_eps_buf)
    var ro0 = TRAINER._readout_loss[BATCH](mu_eps_buf)

    for _ in range(t_infer):
        TRAINER._inference_step[BATCH](
            x_batch, y_batch, params, latents, mu_eps_buf,
            a_below_buf, z_below_buf, dx_buf, Scalar[dtype](lr_x),
        )
    TRAINER._forward_eps[BATCH](
        x_batch, y_batch, params, latents, mu_eps_buf, a_below_buf
    )
    var e1 = TRAINER._total_energy[BATCH](mu_eps_buf)
    var ro1 = TRAINER._readout_loss[BATCH](mu_eps_buf)

    print("  ──", tag)
    print("     energy ", fit(String(e0), 11), "→", fit(String(e1), 11),
          "  readout", fit(String(ro0), 9), "→", fit(String(ro1), 9))
    _print_profile(String("   t=") + String(t_infer), eps_raw)


def main() raises:
    print("=" * 78)
    print("PC per-level error profile — 6-level P10 stack (NO norm levels)")
    print("=" * 78)
    print("  levels (l = 0 bottom .. ", NET.N - 1, " readout):")
    print("    0 conv3->32    1 conv32->64   2 conv64->128")
    print("    3 conv128->128 4 fc2048->512  5 readout512->10")
    print("  T_INFER =", T_INFER, " BATCH =", BATCH)
    print("\n  r_l = rms(eps_{l+1}) / rms(eps_l).  r >> 1 at every rung and")
    print("  roughly CONSTANT = exponential decay downward (both papers'")
    print("  premise).  r ~ 1 = error reaches the bottom; neither mechanism")
    print("  is our bottleneck.")

    var ds = CIFAR10()

    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    unsafe_memset(params_buf, 0, NET.PARAM_SIZE)
    unsafe_memset(grads_buf, 0, NET.PARAM_SIZE)
    var params = tile[NET.PARAM_SIZE](params_buf)
    var grads = tile[NET.PARAM_SIZE](grads_buf)
    NET.pc_init_params[PCXavier, dtype](params)

    var opt_state_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE * OPT.STATE_PER_PARAM).as_unsafe_any_origin()
    var opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    unsafe_memset(opt_state_buf, 0, NET.PARAM_SIZE * OPT.STATE_PER_PARAM)
    unsafe_memset(opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
    var opt_state = tile[NET.PARAM_SIZE, OPT.STATE_PER_PARAM](opt_state_buf)
    var opt_global = tile[OPT.GLOBAL_STATE_SIZE](opt_global_buf)

    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_eps_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var a_below_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var z_below_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var x_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var y_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    unsafe_memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
    unsafe_memset(mu_eps_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
    unsafe_memset(a_below_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(z_below_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(dx_raw, 0, BATCH * NET.LATENT_DIM)

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_raw)
    var x_batch = tile[BATCH, IN](x_buf)
    var y_batch = tile[BATCH, NET.OUT_DIM](y_buf)

    # Train ONCE at the recipe's own settings, then sweep the relaxation.
    print("\n  training", WARM_STEPS, "steps at the P10 recipe (lr_x =",
          LR_X, ", T =", T_INFER, ") ...")
    var step_num: Int = 0
    for b in range(WARM_STEPS):
        for i in range(BATCH):
            var sidx = b * BATCH + i
            for j in range(IN):
                x_buf[i * IN + j] = ds.train_images[sidx * IN + j]
            for c in range(NET.OUT_DIM):
                y_buf[i * NET.OUT_DIM + c] = 0
            y_buf[i * NET.OUT_DIM + Int(ds.train_labels[sidx])] = 1
        var r = TRAINER.compute_grads_only[BATCH](
            params, grads, latents, mu_eps_buf, a_below_buf,
            z_below_buf, dx_buf, x_batch, y_batch,
            T_infer=T_INFER, lr_x=Scalar[dtype](LR_X),
        )
        step_num += 1
        OPT.step[NET.PARAM_SIZE, dtype](params, grads, opt_state, opt_global, step_num)
        if b == WARM_STEPS - 1:
            print("    last sup_loss:", r.output_loss_final)

    # Fixed probe batch for every arm.
    for i in range(BATCH):
        for j in range(IN):
            x_buf[i * IN + j] = ds.train_images[i * IN + j]
        for c in range(NET.OUT_DIM):
            y_buf[i * NET.OUT_DIM + c] = 0
        y_buf[i * NET.OUT_DIM + Int(ds.train_labels[i])] = 1

    var lrs: List[Float64] = [0.025, 0.005, 0.001]
    var ts = [20, 100]

    print("\n" + "=" * 78)
    print("RELAXATION SWEEP — same weights, same batch, only the x-loop varies")
    print("=" * 78)
    print("           level:        0         1         2         3         4         5")
    for li in range(len(lrs)):
        for ti in range(len(ts)):
            _probe(
                String("lr_x=") + String(lrs[li]) + "  T=" + String(ts[ti]),
                lrs[li], ts[ti], mu_eps_raw, x_batch, y_batch, params,
                latents, mu_eps_buf, a_below_buf, z_below_buf, dx_buf,
            )

    print("\n" + "=" * 78)
    print("Bottom pool gone AND energy descends like the baseline => lr_x was the wall.")
    print("Bottom pool persists at every lr_x                     => structural, not step size.")
    print("Bottom pool gone but energy stalls                     => vacuous arm.")
