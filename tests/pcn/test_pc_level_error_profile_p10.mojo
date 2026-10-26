"""Per-level error profile on the P10 net — the norm-free control.

Companion to `test_pc_level_error_profile.mojo`, which probes the 8-level P6
stack and finds a NON-MONOTONE profile: a clean geometric top-down decay over
the upper levels, and a separate, far larger error pool at the bottom three.
Three of that net's eight levels are RMSNorm PC levels, and P8 already found
that RMSNorm makes mu scale-invariant in x_below so "the prediction no longer
pins the latent scale" — a prime suspect for the bottom pool.

This is the control: the 6-level P10 net, 4 convs + 2 FC, NO normalization
anywhere. If the bottom pool disappears here, it is the norm levels; if it
survives, it is the conv stack itself.

Per-level error profile — does our sPC actually suffer signal decay?

DIAGNOSTIC, not a pass/fail gate. P11–P13 tested two cures (Qi et al.'s
spiking schedule and forward updates) without first checking whether our net
has the disease they treat. This measures the disease directly.

Both papers make a claim about the SHAPE of ε across levels:

  * Qi et al. 2025 (arXiv:2506.23800): deep PCNs degrade because of
    "exponentially imbalanced errors between layers during weight updates".
  * Goemaere et al. 2026 (arXiv:2505.20137, ePC): state-based PC suffers
    "exponential signal decay" in digital simulation — the output error fails
    to reach the lower levels, which is what stalls deep sPC.

Both predict the SAME observable: ‖ε_l‖ falling geometrically as l decreases
(away from the readout, where the loss injects error). A roughly CONSTANT
adjacent-level ratio r = rms(ε_{l+1})/rms(ε_l) with r ≫ 1 is exponential
decay, and its size is the decay rate per level. r ≈ 1 means the error
reaches the bottom of the stack fine, and NEITHER paper's mechanism is our
bottleneck.

What we measure, on the 8-level P6 stack (our deepest):
  * rms(ε_l) per level at inference steps t = 1, T/2, T
  * the adjacent-level ratios, which is where "exponential" lives
  * both at INIT and after one epoch of real training, so we can tell an
    initialization artifact from a persistent property

Sanity anchor (printed, non-vacuous): straight after the forward sweep
`init_latents` sets x_l ← μ_l, so every INTERIOR ε must be ~0 and only the
readout carries error. If that is not what we see, the probe is reading the
wrong slabs and every number below is meaningless.

Run:
    pixi run mojo run -I . tests/pcn/test_pc_level_error_profile.mojo
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
    """Walk the relaxation by hand, printing the profile at t=0,1,T/2,T."""
    NET.init_latents[BATCH, dtype](x_batch, params, latents)
    TRAINER._forward_eps[BATCH](
        x_batch, y_batch, params, latents, mu_eps_buf, a_below_buf
    )
    # Anchor: x_l was just set to mu_l, so interior eps must be ~0.
    _print_profile(tag + " t=0", eps_raw)

    for t in range(T_INFER):
        TRAINER._inference_step[BATCH](
            x_batch, y_batch, params, latents, mu_eps_buf,
            a_below_buf, z_below_buf, dx_buf, Scalar[dtype](LR_X),
        )
        if t == 0 or t == T_INFER // 2 - 1:
            TRAINER._forward_eps[BATCH](
                x_batch, y_batch, params, latents, mu_eps_buf, a_below_buf
            )
            _print_profile(tag + " t=" + String(t + 1), eps_raw)
    TRAINER._forward_eps[BATCH](
        x_batch, y_batch, params, latents, mu_eps_buf, a_below_buf
    )
    _print_profile(tag + " t=" + String(T_INFER), eps_raw)


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

    print("\n  ── AT INITIALIZATION (Xavier) ──")
    print("           level:        0         1         2         3         4         5")
    for i in range(BATCH):
        for j in range(IN):
            x_buf[i * IN + j] = ds.train_images[i * IN + j]
        for c in range(NET.OUT_DIM):
            y_buf[i * NET.OUT_DIM + c] = 0
        y_buf[i * NET.OUT_DIM + Int(ds.train_labels[i])] = 1
    _probe(String("init"), mu_eps_raw, x_batch, y_batch, params, latents,
           mu_eps_buf, a_below_buf, z_below_buf, dx_buf)

    print("\n  ── training", WARM_STEPS, "steps (plain PC, Adam 1e-3) ──")
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

    print("\n  ── AFTER", WARM_STEPS, "STEPS ──")
    print("           level:        0         1         2         3         4         5")
    for i in range(BATCH):
        for j in range(IN):
            x_buf[i * IN + j] = ds.train_images[i * IN + j]
        for c in range(NET.OUT_DIM):
            y_buf[i * NET.OUT_DIM + c] = 0
        y_buf[i * NET.OUT_DIM + Int(ds.train_labels[i])] = 1
    _probe(String("trained"), mu_eps_raw, x_batch, y_batch, params, latents,
           mu_eps_buf, a_below_buf, z_below_buf, dx_buf)

    print("\n" + "=" * 78)
    print("Read: constant r >> 1 across rungs => exponential decay (both papers).")
    print("      r ~ 1 => error reaches the bottom; look elsewhere for the wall.")
