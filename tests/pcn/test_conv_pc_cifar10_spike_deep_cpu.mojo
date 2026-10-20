"""Conv-PCN CIFAR-10 — spiking precision on the DEEP (8-level) net, CPU.

The P10 screen (`test_conv_pc_cifar10_spike_sweep_cpu.mojo`) found the spike
inert on the 6-level net. Qi et al. 2025 (arXiv:2506.23800) locate the problem
they solve at DEPTH — "significant performance degradation beyond five to
seven layers" from exponentially imbalanced inter-layer errors. Six levels may
simply not have the disease.

This runs the same schedule on the P6 stack: RMSNorm PC levels interleaved
between convs, **8 levels**, which is our own documented depth failure — it
plateaus ~0.38, BELOW the 5-level unnormalized 0.465, and PCN_CONV_DESIGN.md
attributes that to "PC inference at fixed T_INFER under-settles the deeper
chain". That is Qi et al.'s diagnosis in our words, so this is the honest test
of their fix.

Arms are a 3×2 factorial in σ × forward-updates at FIXED T=12, so neither
axis confounds the other:

    σ ∈ {1.0, 0.5, 0.25}  ×  F ∈ {off, on}

F is Qi et al.'s Fix 2: the weight gradient is taken against the initial
feedforward prediction (ε̃ = x_T − μ_0) instead of the settled one. They report
that spiking ALONE suffices only for iPC — plain PC needs S+F. Arms A/B/C are
S-only (the earlier negative screen), D/E/F add forward updates, and D is F
alone. The paper's headline combination is arm E or F.

Recipe matches P6 exactly (`test_conv_pc_cifar10_norm_cpu.mojo`): plain Adam
at 1e-3, no grad clipping, no weight decay, no h-flip. Budget is reduced to a
screen; arm A is the in-process reference, NOT the recorded 0.38.

Run:
    pixi run mojo run -I . tests/pcn/test_conv_pc_cifar10_spike_deep_cpu.mojo
"""

from std.memory import alloc, unsafe_memset
from std.time import perf_counter_ns
from noeira.core.tiles import tile

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
from noeira.experimental.pcn.pc_trainer import PCTrainResult
from noeira.experimental.pcn.pc_conv_block import ConvPCBlock
from noeira.experimental.pcn.pc_norm_block import NormPCBlock
from noeira.core.fmt import fit

comptime BATCH = 125
comptime EPOCHS = 3                    # SCREEN (P6 recipe: 5)
comptime N_TRAIN_SAMPLES = 10000       # SCREEN (P6 recipe: 20000)
comptime LR_X: Float32 = 0.05
comptime ADAM_LR: Float64 = 0.001
comptime N_TEST_SAMPLES = 1000
comptime N_TRAIN_BATCHES = N_TRAIN_SAMPLES // BATCH
comptime N_TEST_BATCHES = N_TEST_SAMPLES // BATCH
comptime IN = 3 * 32 * 32

comptime NET = PCSequential[
    ConvPCBlock[3, 32, 3, 2, 1, 32, 32, PCIdentity],
    NormPCBlock[8192],
    ConvPCBlock[32, 64, 3, 2, 1, 16, 16, PCReLU],
    NormPCBlock[4096],
    ConvPCBlock[64, 64, 3, 2, 1, 8, 8, PCReLU],
    NormPCBlock[1024],
    PCBlock[1024, 256, PCReLU],
    PCBlock[256, 10, PCIdentity],
]
comptime TRAINER = PCTrainer[
    ConvPCBlock[3, 32, 3, 2, 1, 32, 32, PCIdentity],
    NormPCBlock[8192],
    ConvPCBlock[32, 64, 3, 2, 1, 16, 16, PCReLU],
    NormPCBlock[4096],
    ConvPCBlock[64, 64, 3, 2, 1, 8, 8, PCReLU],
    NormPCBlock[1024],
    PCBlock[1024, 256, PCReLU],
    PCBlock[256, 10, PCIdentity],
    dtype=dtype,
]
comptime OPT = PCAdam[LR=ADAM_LR]


def main() raises:
    print("=" * 72)
    print("Conv-PCN CIFAR-10 — spiking precision on the DEEP net (CPU)")
    print("=" * 72)
    print("  net       : 8 levels (P6 normalized stack), N =", NET.N)
    print("  budget    :", N_TRAIN_SAMPLES, "train /", EPOCHS, "ep  (SCREEN)")
    print("  PARAM_SIZE=", NET.PARAM_SIZE, " LATENT_DIM=", NET.LATENT_DIM)

    var ds = CIFAR10()

    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var snap_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    unsafe_memset(params_buf, 0, NET.PARAM_SIZE)
    unsafe_memset(snap_buf, 0, NET.PARAM_SIZE)
    unsafe_memset(grads_buf, 0, NET.PARAM_SIZE)
    var params = tile[NET.PARAM_SIZE](params_buf)
    var grads = tile[NET.PARAM_SIZE](grads_buf)

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
    var lat0_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var eval_out_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    var x_buf = alloc[Scalar[dtype]](BATCH * IN).as_unsafe_any_origin()
    var y_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_buf_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_buf_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_buf_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_buf_raw)
    var latents_0 = tile[BATCH, NET.LATENT_DIM](lat0_buf)
    var eval_out = tile[BATCH, NET.OUT_DIM](eval_out_buf)
    var x_batch = tile[BATCH, IN](x_buf)
    var y_batch = tile[BATCH, NET.OUT_DIM](y_buf)

    comptime T_FIXED = 12
    var arm_S: List[Float64] = [1.0, 0.5, 0.25, 1.0, 0.5, 0.25]
    var arm_F = [False, False, False, True, True, True]
    var arm_name = [
        String("A  σ=1.00  F=off  (P6 ref)"),
        String("B  σ=0.50  F=off  (S)"),
        String("C  σ=0.25  F=off  (S)"),
        String("D  σ=1.00  F=on   (F)"),
        String("E  σ=0.50  F=on   (S+F)"),
        String("F  σ=0.25  F=on   (S+F)"),
    ]
    var results: List[Float64] = []
    var losses: List[Float64] = []
    var walls: List[Float64] = []

    for a in range(len(arm_S)):
        var T_infer = T_FIXED
        var sigma = Scalar[dtype](arm_S[a])
        var use_fwd = arm_F[a]

        for i in range(NET.PARAM_SIZE):
            params_buf[i] = snap_buf[i]
        unsafe_memset(grads_buf, 0, NET.PARAM_SIZE)
        unsafe_memset(opt_state_buf, 0, NET.PARAM_SIZE * OPT.STATE_PER_PARAM)
        unsafe_memset(opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)
        unsafe_memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
        unsafe_memset(mu_eps_buf_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
        unsafe_memset(a_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
        unsafe_memset(z_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
        unsafe_memset(dx_buf_raw, 0, BATCH * NET.LATENT_DIM)
        unsafe_memset(lat0_buf, 0, BATCH * NET.LATENT_DIM)

        var step_num: Int = 0
        var best_acc: Float64 = 0.0
        var last_loss: Float64 = 0.0
        var t0 = perf_counter_ns()

        print("\n  ── arm", arm_name[a], "──")
        print("     epoch | sup_loss  | test_acc | best   | wall_t (s)")
        print("     ------+-----------+----------+--------+-----------")

        for epoch in range(EPOCHS):
            var ep_loss: Float64 = 0.0
            for batch_idx in range(N_TRAIN_BATCHES):
                for i in range(BATCH):
                    var sidx = batch_idx * BATCH + i
                    for j in range(IN):
                        x_buf[i * IN + j] = ds.train_images[sidx * IN + j]
                    for c in range(NET.OUT_DIM):
                        y_buf[i * NET.OUT_DIM + c] = 0
                    y_buf[i * NET.OUT_DIM + Int(ds.train_labels[sidx])] = 1

                var r: PCTrainResult
                if use_fwd:
                    r = TRAINER.compute_grads_only_fwd[BATCH](
                        params, grads, latents, latents_0, mu_eps_buf,
                        a_below_buf, z_below_buf, dx_buf, x_batch, y_batch,
                        T_infer=T_infer, lr_x=Scalar[dtype](LR_X),
                        spike_sigma=sigma,
                    )
                else:
                    r = TRAINER.compute_grads_only[BATCH](
                        params, grads, latents, mu_eps_buf, a_below_buf,
                        z_below_buf, dx_buf, x_batch, y_batch,
                        T_infer=T_infer, lr_x=Scalar[dtype](LR_X),
                        spike_sigma=sigma,
                    )
                ep_loss += r.output_loss_final
                step_num += 1
                OPT.step[NET.PARAM_SIZE, dtype](params, grads, opt_state, opt_global, step_num)

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
            last_loss = ep_loss / Float64(N_TRAIN_BATCHES)
            var el = Float64(perf_counter_ns() - t0) / 1e9
            print("       ", epoch, "  ", fit(String(last_loss), 9),
                  "  ", fit(String(acc), 8), "  ", fit(String(best_acc), 6),
                  "  ", el)

        results.append(best_acc)
        losses.append(last_loss)
        walls.append(Float64(perf_counter_ns() - t0) / 1e9)

    print("\n" + "=" * 72)
    print("DEEP SCREEN SUMMARY — 8 levels, T =", T_FIXED, ",",
          N_TRAIN_SAMPLES, "train /", EPOCHS, "ep")
    print("=" * 72)
    print("  arm                       | best_acc | Δ vs A   | sup_loss | wall (s)")
    print("  --------------------------+----------+----------+----------+---------")
    for a in range(len(results)):
        print("  ", fit(arm_name[a], 24), " ", fit(String(results[a]), 8),
              " ", fit(String(results[a] - results[0]), 8),
              " ", fit(String(losses[a]), 8),
              " ", fit(String(walls[a]), 7))
    print("\n  Arm A is the P6 recipe at screen budget. The recorded P6 plateau")
    print("  (0.38 @ 8 ep / 20k) is NOT comparable to these numbers.")
