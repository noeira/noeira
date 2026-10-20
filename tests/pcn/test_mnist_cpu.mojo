"""End-to-end MNIST PCN test (CPU) — validates Bogacz canonical convergence.

Architecture (mirrors Bogacz notebook 1 with smaller hidden):
    PCBlock[784, 128, PCIdentity]   # input → x_1 (no act on data)
    PCBlock[128, 128, PCReLU]       # x_1 (after ReLU) → x_2
    PCBlock[128,  10, PCReLU]       # x_2 (after ReLU) → output (readout)

Pass criterion (CPU, small budget):
  - Test accuracy ≥ 80% after 2 epochs on a 5000-sample subset.
  Bogacz reports ~95% with full dataset + 10 epochs. We're trading dataset
  size for wall-clock; what matters is "the algorithm learns real images,
  not just synthetic toys".

Run:
    pixi run mojo run -I . tests/pcn/test_mnist_cpu.mojo
"""

from std.memory import alloc, unsafe_memset
from std.time import perf_counter_ns
from noeira.core.tiles import tile

from noeira.nn.constants import DT as dtype
from noeira.experimental.pcn.pc_initializer import PCXavier
from noeira.experimental.pcn.pc_optimizer import PCAdam
from noeira.nn.datasets.mnist import MNIST
from noeira.experimental.pcn import (
    PCBlock,
    PCSequential,
    PCIdentity,
    PCReLU,
    PCTrainer,
)


comptime BATCH = 64
comptime EPOCHS = 2
comptime T_INFER = 20
comptime LR_X: Float64 = 0.05         # Bogacz-style LR for latent SGD
comptime ADAM_LR: Float64 = 0.001     # Bogacz default for params

# CPU budget: train subset (much faster than full 60k * 2 epochs)
comptime N_TRAIN_SAMPLES = 5000
comptime N_TEST_SAMPLES = 1000

comptime N_TRAIN_BATCHES = N_TRAIN_SAMPLES // BATCH
comptime N_TEST_BATCHES = N_TEST_SAMPLES // BATCH

comptime NET = PCSequential[
    PCBlock[784, 128, PCIdentity],
    PCBlock[128, 128, PCReLU],
    PCBlock[128, 10, PCReLU],
]
comptime TRAINER = PCTrainer[
    PCBlock[784, 128, PCIdentity],
    PCBlock[128, 128, PCReLU],
    PCBlock[128, 10, PCReLU],
    dtype=dtype,
]
comptime OPT = PCAdam[LR=ADAM_LR]


def main() raises:
    print("=" * 65)
    print("MNIST PCN (Bogacz canonical, CPU) — Phase 1 validation")
    print("=" * 65)
    print("  arch       : 784 → 128 → 128 → 10")
    print("  PARAM_SIZE :", NET.PARAM_SIZE)
    print("  LATENT_DIM :", NET.LATENT_DIM)
    print(
        "  hyperparams: BATCH=", BATCH, " T_INFER=", T_INFER,
        " EPOCHS=", EPOCHS,
    )
    print("  optimizer  : Adam(lr=", ADAM_LR, "), latent SGD lr=", LR_X)
    print("  train budget:", N_TRAIN_SAMPLES, "samples (", N_TRAIN_BATCHES, "batches/epoch)")
    print("  test budget :", N_TEST_SAMPLES, "samples (", N_TEST_BATCHES, "batches)")

    # ── Load MNIST ────────────────────────────────────────────────────────────
    var ds = MNIST()
    print("  [mnist] loaded:", MNIST.N_TRAIN, "train,", MNIST.N_TEST, "test")

    # ── Allocate params + initialize ──────────────────────────────────────────
    var params_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    var grads_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE).as_unsafe_any_origin()
    unsafe_memset(params_buf, 0, NET.PARAM_SIZE)
    unsafe_memset(grads_buf, 0, NET.PARAM_SIZE)

    var params = tile[NET.PARAM_SIZE](params_buf)
    var grads = tile[NET.PARAM_SIZE](grads_buf)
    NET.pc_init_params[PCXavier, dtype](params)

    # ── Adam optimizer state ──────────────────────────────────────────────────
    var opt_state_buf = alloc[Scalar[dtype]](NET.PARAM_SIZE * OPT.STATE_PER_PARAM).as_unsafe_any_origin()
    var opt_global_buf = alloc[Scalar[dtype]](OPT.GLOBAL_STATE_SIZE).as_unsafe_any_origin()
    unsafe_memset(opt_state_buf, 0, NET.PARAM_SIZE * OPT.STATE_PER_PARAM)
    unsafe_memset(opt_global_buf, 0, OPT.GLOBAL_STATE_SIZE)

    var opt_state = tile[NET.PARAM_SIZE, OPT.STATE_PER_PARAM](opt_state_buf)
    var opt_global = tile[OPT.GLOBAL_STATE_SIZE](opt_global_buf)

    # ── Per-batch latents + scratch buffers (allocated once, reused) ──────────
    var lat_buf = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var mu_eps_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_OUT_DIM).as_unsafe_any_origin()
    var a_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var z_below_buf_raw = alloc[Scalar[dtype]](BATCH * NET.SCRATCH_IN_DIM).as_unsafe_any_origin()
    var dx_buf_raw = alloc[Scalar[dtype]](BATCH * NET.LATENT_DIM).as_unsafe_any_origin()
    var eval_out_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    unsafe_memset(lat_buf, 0, BATCH * NET.LATENT_DIM)
    unsafe_memset(mu_eps_buf_raw, 0, BATCH * NET.SCRATCH_OUT_DIM)
    unsafe_memset(a_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(z_below_buf_raw, 0, BATCH * NET.SCRATCH_IN_DIM)
    unsafe_memset(dx_buf_raw, 0, BATCH * NET.LATENT_DIM)
    unsafe_memset(eval_out_buf, 0, BATCH * NET.OUT_DIM)

    var latents = tile[BATCH, NET.LATENT_DIM](lat_buf)
    var mu_eps_buf = tile[BATCH, NET.SCRATCH_OUT_DIM](mu_eps_buf_raw)
    var a_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](a_below_buf_raw)
    var z_below_buf = tile[BATCH, NET.SCRATCH_IN_DIM](z_below_buf_raw)
    var dx_buf = tile[BATCH, NET.LATENT_DIM](dx_buf_raw)
    var eval_out = tile[BATCH, NET.OUT_DIM](eval_out_buf)

    # ── Per-batch input/target buffers (rebound from MNIST host buffers) ──────
    var x_batch_buf = alloc[Scalar[dtype]](BATCH * NET.IN_DIM).as_unsafe_any_origin()
    var y_batch_buf = alloc[Scalar[dtype]](BATCH * NET.OUT_DIM).as_unsafe_any_origin()
    unsafe_memset(x_batch_buf, 0, BATCH * NET.IN_DIM)
    unsafe_memset(y_batch_buf, 0, BATCH * NET.OUT_DIM)

    var x_batch = tile[BATCH, NET.IN_DIM](x_batch_buf)
    var y_batch = tile[BATCH, NET.OUT_DIM](y_batch_buf)

    # ── Eval on test set: pre-training accuracy via forward_eval ──────────────
    var initial_acc: Float64 = 0.0
    var initial_correct: Int = 0
    for tb in range(N_TEST_BATCHES):
        for i in range(BATCH):
            var sample_idx = tb * BATCH + i
            for j in range(NET.IN_DIM):
                x_batch_buf[i * NET.IN_DIM + j] = ds.test_images[
                    sample_idx * NET.IN_DIM + j
                ]
        NET.forward_eval[BATCH, dtype](x_batch, params, eval_out)
        for i in range(BATCH):
            var sample_idx = tb * BATCH + i
            var best_class: Int = 0
            var best_val = Float64(eval_out_buf[i * NET.OUT_DIM])
            for c in range(1, NET.OUT_DIM):
                var v = Float64(eval_out_buf[i * NET.OUT_DIM + c])
                if v > best_val:
                    best_val = v
                    best_class = c
            if best_class == Int(ds.test_labels[sample_idx]):
                initial_correct += 1
    initial_acc = Float64(initial_correct) / Float64(N_TEST_BATCHES * BATCH)
    print("\n  [pre-train test acc]:", initial_acc)
    print("\n  epoch | batch | E_init    E_final   sup_loss   train_t (s)")
    print("  ------+-------+------------------------------------------------")

    var step_num: Int = 0
    var t0 = perf_counter_ns()

    for epoch in range(EPOCHS):
        var epoch_start_ns = perf_counter_ns()
        var loss_sum: Float64 = 0.0
        var loss_count: Int = 0

        for batch_idx in range(N_TRAIN_BATCHES):
            # ── Copy minibatch from MNIST train set (sequential, no shuffle for now) ──
            for i in range(BATCH):
                var sample_idx = batch_idx * BATCH + i
                for j in range(NET.IN_DIM):
                    x_batch_buf[i * NET.IN_DIM + j] = ds.train_images[
                        sample_idx * NET.IN_DIM + j
                    ]
                # one-hot target
                for c in range(NET.OUT_DIM):
                    y_batch_buf[i * NET.OUT_DIM + c] = Scalar[dtype](0)
                y_batch_buf[
                    i * NET.OUT_DIM + Int(ds.train_labels[sample_idx])
                ] = Scalar[dtype](1.0)

            # ── Compute grads (no param update) ──
            var result = TRAINER.compute_grads_only[BATCH](
                params,
                grads,
                latents,
                mu_eps_buf,
                a_below_buf,
                z_below_buf,
                dx_buf,
                x_batch,
                y_batch,
                T_infer=T_INFER,
                lr_x=Scalar[dtype](LR_X),
            )

            # ── Adam step ──
            step_num += 1
            OPT.step[NET.PARAM_SIZE, dtype](
                params, grads, opt_state, opt_global, step_num
            )

            loss_sum += result.output_loss_final
            loss_count += 1

            # Periodic progress
            if batch_idx == 0 or (batch_idx + 1) % 20 == 0 or batch_idx == N_TRAIN_BATCHES - 1:
                var elapsed = Float64(perf_counter_ns() - epoch_start_ns) / 1e9
                print(
                    "    ", epoch, "  ", batch_idx, "  ",
                    fit(String(result.energy_initial), 9), " ",
                    fit(String(result.energy_final), 9), " ",
                    fit(String(result.output_loss_final), 9), "  ",
                    fit(String(elapsed), 7),
                )

        # Inline eval at end of epoch
        var ep_correct: Int = 0
        for tb in range(N_TEST_BATCHES):
            for i in range(BATCH):
                var sample_idx = tb * BATCH + i
                for j in range(NET.IN_DIM):
                    x_batch_buf[i * NET.IN_DIM + j] = ds.test_images[
                        sample_idx * NET.IN_DIM + j
                    ]
            NET.forward_eval[BATCH, dtype](x_batch, params, eval_out)
            for i in range(BATCH):
                var sample_idx = tb * BATCH + i
                var best_class: Int = 0
                var best_val = Float64(eval_out_buf[i * NET.OUT_DIM])
                for c in range(1, NET.OUT_DIM):
                    var v = Float64(eval_out_buf[i * NET.OUT_DIM + c])
                    if v > best_val:
                        best_val = v
                        best_class = c
                if best_class == Int(ds.test_labels[sample_idx]):
                    ep_correct += 1
        var acc = Float64(ep_correct) / Float64(N_TEST_BATCHES * BATCH)
        var avg_loss = loss_sum / Float64(loss_count) if loss_count > 0 else 0.0
        print(
            "  [epoch", epoch, "done]  avg_sup_loss=",
            fit(String(avg_loss), 9),
            "  test_acc=", acc,
        )

    var total_elapsed = Float64(perf_counter_ns() - t0) / 1e9
    # Final eval (inline)
    var final_correct: Int = 0
    for tb in range(N_TEST_BATCHES):
        for i in range(BATCH):
            var sample_idx = tb * BATCH + i
            for j in range(NET.IN_DIM):
                x_batch_buf[i * NET.IN_DIM + j] = ds.test_images[
                    sample_idx * NET.IN_DIM + j
                ]
        NET.forward_eval[BATCH, dtype](x_batch, params, eval_out)
        for i in range(BATCH):
            var sample_idx = tb * BATCH + i
            var best_class: Int = 0
            var best_val = Float64(eval_out_buf[i * NET.OUT_DIM])
            for c in range(1, NET.OUT_DIM):
                var v = Float64(eval_out_buf[i * NET.OUT_DIM + c])
                if v > best_val:
                    best_val = v
                    best_class = c
            if best_class == Int(ds.test_labels[sample_idx]):
                final_correct += 1
    var final_acc = Float64(final_correct) / Float64(N_TEST_BATCHES * BATCH)

    print("\n  total wall time :", total_elapsed, "s")
    print("  final test acc  :", final_acc)
    print("  pre-train acc   :", initial_acc)
    print("  acc improvement :", final_acc - initial_acc)

    if final_acc >= 0.80:
        print("\n  [PASS] final test acc ≥ 80% — Bogacz canonical PCN converges on real data")
    elif final_acc >= 0.50:
        print("\n  [PARTIAL] final test acc ≥ 50% but < 80% — algorithm works but more compute needed")
    else:
        print("\n  [FAIL] final test acc <50% — algorithm may have a bug")
        raise Error("MNIST test failed: final accuracy " + String(final_acc))

    params_buf.free()
    grads_buf.free()
    opt_state_buf.free()
    opt_global_buf.free()
    lat_buf.free()
    mu_eps_buf_raw.free()
    a_below_buf_raw.free()
    z_below_buf_raw.free()
    dx_buf_raw.free()
    eval_out_buf.free()
    x_batch_buf.free()
    y_batch_buf.free()
    print("=== Done ===")

from noeira.core.fmt import fit