"""ConvPCBlock end-to-end PC training (P1, CPU) — see docs/PCN_CONV_DESIGN.md.

Drops a real ConvPCBlock into PCSequential + PCTrainer and runs the genuine
Bogacz-canonical training loop on a fixed synthetic batch. Two claims:

  1. The inference loop reduces energy every step: E_final < E_initial.
  2. The local weight rule learns: the readout loss drops substantially as the
     fixed batch is overfit.

Network (input conv → MLP readout, all flat-composed, no Flatten op):
  ConvPCBlock[1, 4, 3, 1, 1, 6, 6, PCIdentity]   # 1×6×6 (36) → 4×6×6 (144)
  PCBlock[144, 32, PCReLU]                        # 144 → 32
  PCBlock[32, 8, PCIdentity]                      # 32 → 8 readout

Run:
    pixi run mojo run -I . tests/pcn/test_conv_pc_end_to_end_cpu.mojo
"""

from std.memory import alloc, memset
from std.math import sin
from noeira.core.tiles import tile

from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCReLU,
    PCTrainer,
)
from noeira.experimental.pcn.pc_conv_block import ConvPCBlock
from noeira.core.fmt import fit

comptime dtype = DType.float32
comptime BATCH = 4
comptime STEPS = 200
comptime T_INFER = 20
comptime LR_X: Float32 = 0.1
comptime LR_W: Float32 = 0.02

comptime NET = PCSequential[
    ConvPCBlock[1, 4, 3, 1, 1, 6, 6, PCIdentity],
    PCBlock[144, 32, PCReLU],
    PCBlock[32, 8, PCIdentity],
]
comptime TRAINER = PCTrainer[
    ConvPCBlock[1, 4, 3, 1, 1, 6, 6, PCIdentity],
    PCBlock[144, 32, PCReLU],
    PCBlock[32, 8, PCIdentity],
    dtype=dtype,
]


def main() raises:
    print("ConvPCBlock end-to-end PC training (P1, CPU)\n")
    print("  IN_DIM=", NET.IN_DIM, " OUT_DIM=", NET.OUT_DIM,
          " LATENT_DIM=", NET.LATENT_DIM, " PARAM_SIZE=", NET.PARAM_SIZE)

    # ── Params + grads ────────────────────────────────────────────────────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    memset(params_buf, 0, NET.PARAM_SIZE)
    memset(grads_buf, 0, NET.PARAM_SIZE)
    var params = tile[NET.PARAM_SIZE](params_buf)
    var grads = tile[NET.PARAM_SIZE](grads_buf)
    NET.pc_init_params[PCXavier, dtype](params)

    # ── Per-batch latents + scratch ──────────────────────────────────────────
    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_eps_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var a_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var z_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
    memset(mu_eps_buf_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
    memset(a_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(z_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    memset(dx_buf_raw, 0, BATCH * NET.LATENT_DIM)

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_buf_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_buf_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_buf_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_buf_raw)

    # ── Fixed synthetic batch (deterministic) ────────────────────────────────
    var x_buf = alloc[Scalar[dtype]](BATCH * NET.IN_DIM).as_unsafe_any_origin()
    var y_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    for i in range(BATCH * NET.IN_DIM):
        x_buf[i] = Scalar[dtype](sin(Float32(i) * 0.5 + 0.1))
    for i in range(BATCH * NET.OUT_DIM):
        y_buf[i] = Scalar[dtype](0.5 * sin(Float32(i) * 0.9 + 0.4))
    var x_in = tile[BATCH, NET.IN_DIM](x_buf)
    var y_target = tile[BATCH, NET.OUT_DIM](y_buf)

    # ── Train ─────────────────────────────────────────────────────────────────
    var first_loss: Float64 = 0.0
    var last_loss: Float64 = 0.0
    var energy_descent_ok = True
    var violations: Int = 0

    print("\n  step |  E_init     E_final    sup_loss")
    print("  -----+----------------------------------")
    for step in range(STEPS):
        var r = TRAINER.train_one_batch[BATCH](
            params,
            grads,
            latents,
            mu_eps_buf,
            a_below_buf,
            z_below_buf,
            dx_buf,
            x_in,
            y_target,
            T_infer=T_INFER,
            lr_x=Scalar[dtype](LR_X),
            lr_w=Scalar[dtype](LR_W),
        )
        # Inference must not increase energy (allow tiny fp slack).
        if r.energy_final > r.energy_initial + 1e-4:
            energy_descent_ok = False
            violations += 1
        if step == 0:
            first_loss = r.output_loss_final
        last_loss = r.output_loss_final
        if step == 0 or (step + 1) % 40 == 0 or step == STEPS - 1:
            print("    ", step, " ",
                  fit(String(r.energy_initial), 9), " ",
                  fit(String(r.energy_final), 9), " ",
                  fit(String(r.output_loss_final), 9))

    print("\n  first sup_loss =", first_loss)
    print("  last  sup_loss =", last_loss)
    print("  energy-descent violations =", violations, "/", STEPS)

    var learned = last_loss < first_loss * 0.5
    var ok = energy_descent_ok and learned

    print("")
    if ok:
        print("✅ PASS — inference reduces energy every step; readout loss",
              "dropped to", fit(String(last_loss / first_loss * 100.0), 5),
              "% of initial")
    else:
        if not energy_descent_ok:
            print("❌ FAIL — energy increased on", violations, "step(s)")
        if not learned:
            print("❌ FAIL — readout loss did not drop ≥50% (no learning)")
        raise Error("ConvPCBlock end-to-end P1 failed")
