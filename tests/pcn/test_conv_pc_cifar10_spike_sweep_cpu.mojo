"""Conv-PCN CIFAR-10 — spiking-precision SCREEN (CPU, short budget).

Screens the Qi et al. 2025 (arXiv:2506.23800) spiking-precision schedule on
the P10 net at a reduced budget, to decide whether a full 15-epoch/50k run is
worth ~1 h. Same net, same optimizer, same clipping, same data order as
`test_conv_pc_cifar10_bigger_cpu.mojo` — ONLY `T_INFER` and `spike_sigma`
differ between arms.

Discipline (cf. `feedback_the_gates_name_named_the_wrong_axis` and
`_a_baseline_from_earlier_in_the_session_has_drifted`):
  * every arm runs in THIS process, against the recorded 0.6565 recipe's own
    baseline arm — not against a number from an earlier session;
  * all arms restore a byte-identical parameter snapshot, so the init is not
    a confound (`PCXavier` draws from the global RNG — re-calling it would
    give each arm different weights);
  * arms differ on ONE axis at a time: A→B→C vary σ at fixed T; A→D varies T
    at fixed σ; E is the combined cell.

α = `spike_sigma` is the Σ at the spiking level; α < 1 BOOSTS that level's
error. σ = 1 is the disabled path and must reproduce the plain recipe.

⚠ This is a SCREEN, not a result: 2 epochs on 10k is 4% of the P10 budget,
and a short-budget ordering need not survive to full budget. Its job is to
say "worth a full run" or "not", nothing more.

Run:
    pixi run mojo run -I . tests/pcn/test_conv_pc_cifar10_spike_sweep_cpu.mojo
"""

from std.memory import alloc, memset
from std.time import perf_counter_ns
from std.math import sqrt
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.nn.core.ptr import mptr
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
comptime EPOCHS = 2                    # SCREEN budget (P10 recipe: 15)
comptime N_TRAIN_SAMPLES = 10000       # SCREEN budget (P10 recipe: 50000)
comptime LR_X: Float32 = 0.025
comptime ADAM_LR: Float64 = 0.0002
comptime CLIP_FACTOR: Float64 = 5.0
comptime WD: Float64 = 0.0002
comptime N_TEST_SAMPLES = 2000
comptime N_TRAIN_BATCHES = N_TRAIN_SAMPLES // BATCH
comptime N_TEST_BATCHES = N_TEST_SAMPLES // BATCH
comptime IN = 3 * 32 * 32


@always_inline
def _flip_into(
    src: Pointer[Scalar[dtype], MutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
):
    for c in range(3):
        for h in range(32):
            var row = c * 1024 + h * 32
            for w in range(32):
                dst[row + w] = src[row + (31 - w)]


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


def main() raises:
    print("=" * 72)
    print("Conv-PCN CIFAR-10 — spiking-precision SCREEN (CPU)")
    print("=" * 72)
    print("  net       : 6 levels (P10), N =", NET.N)
    print("  budget    :", N_TRAIN_SAMPLES, "train /", EPOCHS, "ep  (SCREEN)")
    print("  PARAM_SIZE=", NET.PARAM_SIZE, " LATENT_DIM=", NET.LATENT_DIM)

    var ds = CIFAR10()

    # ── Buffers: allocated ONCE, shared by every arm ─────────────────────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var snap_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    memset(params_buf, 0, NET.PARAM_SIZE)
    memset(snap_buf, 0, NET.PARAM_SIZE)
    memset(grads_buf, 0, NET.PARAM_SIZE)
    var params = tile[NET.PARAM_SIZE](params_buf)
    var grads = tile[NET.PARAM_SIZE](grads_buf)

    # One draw, snapshotted — every arm restores from this.
    NET.pc_init_params[PCXavier, dtype](params)
    for i in range(NET.PARAM_SIZE):
        snap_buf[i] = params_buf[i]

    var opt_state_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE * OPT.STATE_PER_PARAM).as_unsafe_any_origin()
    var opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    var opt_state = tile[NET.PARAM_SIZE, OPT.STATE_PER_PARAM](opt_state_buf)
    var opt_global = tile[OPT.GLOBAL_STATE_SIZE](opt_global_buf)

    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_eps_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var a_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var z_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var eval_out_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    var x_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var y_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_buf_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_buf_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_buf_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_buf_raw)
    var eval_out = tile[BATCH, NET.OUT_DIM](eval_out_buf)
    var x_batch = tile[BATCH, IN](x_buf)
    var y_batch = tile[BATCH, NET.OUT_DIM](y_buf)

    # ── Arms: (T_infer, spike_sigma) ─────────────────────────────────────────
    var arm_T = [20, 20, 20, 6, 6]
    var arm_S: List[Float64] = [1.0, 0.5, 0.25, 1.0, 0.25]
    var arm_name = [
        String("A  T=20 σ=1.00 (baseline)"),
        String("B  T=20 σ=0.50"),
        String("C  T=20 σ=0.25"),
        String("D  T= 6 σ=1.00"),
        String("E  T= 6 σ=0.25"),
    ]
    var results: List[Float64] = []
    var walls: List[Float64] = []

    for a in range(len(arm_T)):
        var T_infer = arm_T[a]
        var sigma = Scalar[dtype](arm_S[a])

        # Reset EVERYTHING that carries state between arms.
        for i in range(NET.PARAM_SIZE):
            params_buf[i] = snap_buf[i]
        memset(grads_buf, 0, NET.PARAM_SIZE)
        memset(opt_state_buf, 0, NET.PARAM_SIZE * OPT.STATE_PER_PARAM)
        memset(opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
        memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
        memset(mu_eps_buf_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
        memset(a_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
        memset(z_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
        memset(dx_buf_raw, 0, BATCH * NET.LATENT_DIM)

        var step_num: Int = 0
        var best_acc: Float64 = 0.0
        var ema_norm: Float64 = 0.0
        var t0 = perf_counter_ns()

        print("\n  ── arm", arm_name[a], "──")
        print("     epoch | sup_loss  | test_acc | best   | clips | wall_t (s)")
        print("     ------+-----------+----------+--------+-------+-----------")

        for epoch in range(EPOCHS):
            var ep_loss: Float64 = 0.0
            var n_clips: Int = 0
            for batch_idx in range(N_TRAIN_BATCHES):
                for i in range(BATCH):
                    var sidx = batch_idx * BATCH + i
                    var do_flip = ((sidx * 2654435761 + epoch * 40503) >> 13) & 1
                    if do_flip == 1:
                        _flip_into(mptr(ds.train_images).unsafe_offset(sidx * IN), x_buf + i * IN)
                    else:
                        for j in range(IN):
                            x_buf[i * IN + j] = ds.train_images[sidx * IN + j]
                    for c in range(NET.OUT_DIM):
                        y_buf[i * NET.OUT_DIM + c] = 0
                    y_buf[i * NET.OUT_DIM + Int(ds.train_labels[sidx])] = 1

                var r = TRAINER.compute_grads_only[BATCH](
                    params, grads, latents, mu_eps_buf, a_below_buf,
                    z_below_buf, dx_buf, x_batch, y_batch,
                    T_infer=T_infer, lr_x=Scalar[dtype](LR_X),
                    spike_sigma=sigma,
                )
                ep_loss += r.output_loss_final

                var gn: Float64 = 0.0
                for i in range(NET.PARAM_SIZE):
                    var gv = Float64(grads_buf[i])
                    gn += gv * gv
                gn = sqrt(gn)
                if ema_norm > 0.0 and gn > CLIP_FACTOR * ema_norm:
                    var sc = Scalar[dtype]((CLIP_FACTOR * ema_norm) / gn)
                    for i in range(NET.PARAM_SIZE):
                        grads_buf[i] = grads_buf[i] * sc
                    n_clips += 1
                    gn = CLIP_FACTOR * ema_norm
                if ema_norm == 0.0:
                    ema_norm = gn
                else:
                    ema_norm = 0.99 * ema_norm + 0.01 * gn

                step_num += 1
                OPT.step[NET.PARAM_SIZE, dtype](params, grads, opt_state, opt_global, step_num)
                var keep = Scalar[dtype](1.0 - WD)
                for i in range(NET.PARAM_SIZE):
                    params_buf[i] = params_buf[i] * keep

            var correct: Int = 0
            for tb in range(N_TEST_BATCHES):
                for i in range(BATCH):
                    var sidx = tb * BATCH + i
                    for j in range(IN):
                        x_buf[i * IN + j] = ds.test_images[sidx * IN + j]
                NET.forward_eval[BATCH, dtype](x_batch, params, eval_out)
                for i in range(BATCH):
                    var best_c: Int = 0
                    var best_v = Float64(eval_out_buf[i * NET.OUT_DIM])
                    for c in range(1, NET.OUT_DIM):
                        var v = Float64(eval_out_buf[i * NET.OUT_DIM + c])
                        if v > best_v:
                            best_v = v; best_c = c
                    if best_c == Int(ds.test_labels[tb * BATCH + i]):
                        correct += 1
            var acc = Float64(correct) / Float64(N_TEST_BATCHES * BATCH)
            if acc > best_acc:
                best_acc = acc
            var el = Float64(perf_counter_ns() - t0) / 1e9
            print("       ", epoch, "  ", fit(String(ep_loss / Float64(N_TRAIN_BATCHES)), 9),
                  "  ", fit(String(acc), 8), "  ", fit(String(best_acc), 6),
                  "  ", n_clips, "  ", el)

        results.append(best_acc)
        walls.append(Float64(perf_counter_ns() - t0) / 1e9)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SCREEN SUMMARY —", N_TRAIN_SAMPLES, "train /", EPOCHS, "ep")
    print("=" * 72)
    print("  arm                        | best_acc | Δ vs A   | wall (s)")
    print("  ---------------------------+----------+----------+---------")
    for a in range(len(results)):
        print("  ", fit(arm_name[a], 25), " ", fit(String(results[a]), 8),
              " ", fit(String(results[a] - results[0]), 8), " ",
              fit(String(walls[a]), 7))
    print("\n  Baseline arm A is the P10 recipe at screen budget; the recorded")
    print("  full-budget P10 number (0.6565) is NOT comparable to these.")
