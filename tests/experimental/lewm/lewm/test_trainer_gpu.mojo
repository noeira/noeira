"""LeWMTrainer GPU smoke (Phase E, Apple/NVIDIA).

Exercises the device train path end-to-end: Scratch graph IO, the
device loss reduce-accumulate (`read_loss_accum` / `reset_loss_accum`),
the `seed_grad_inv_batch` backward seed, `collapse_probes` D2H of the
`emb` node, and the GPU checkpoint D2H/H2D visitors. The whole JEPA
graph (Tokenwise encoder, ARPredictor, MSEPerSample, SIGReg) runs on the
device.

This is a *finite + decreasing* smoke (small toy config), NOT the §10.7
scale reproduction — that's the NVIDIA run. Mirrors the CPU
`test_trainer.mojo` structure.

Run:  pixi run -e apple mojo run -I . tests/experimental/lewm/test_trainer_gpu.mojo
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import isnan, isinf
from std.testing import assert_true
from layout import TileTensor, row_major

from noeira.nn.constants import DT
from noeira.experimental.lewm.trainer import LeWMTrainer


# toy config (mirrors test_trainer.mojo)
comptime IN_CH = 4
comptime IMG = 8
comptime PATCH = 4
comptime HIDDEN = 8
comptime ENC_HEADS = 2
comptime ENC_LAYERS = 2
comptime EMB = 8
comptime ENC_PROJ_H = 16
comptime ENC_FF_MULT = 2
comptime T = 4
comptime ACT = 3
comptime SMOOTHED = 8
comptime AE_MLP = 2
comptime H = 3
comptime N_PREDS = 1
comptime PRED_HEADS = 2
comptime PRED_FF = 16
comptime DEPTH = 2
comptime PRED_PROJ_H = 16
comptime SIG_PROJ = 8
comptime SIG_KNOTS = 5
comptime B = 4

comptime IMG_DIM = IN_CH * IMG * IMG
comptime PIX = T * IMG_DIM
comptime ACTIN = T * ACT

comptime Trainer = LeWMTrainer[
    IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, ENC_PROJ_H,
    ENC_FF_MULT, T, ACT, SMOOTHED, AE_MLP, H, N_PREDS, PRED_HEADS, PRED_FF,
    DEPTH, PRED_PROJ_H, SIG_PROJ, SIG_KNOTS, B, "gpu",
]


def _p(b: DeviceBuffer[DT]) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return rebind[Pointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


def main() raises:
    print("=" * 70)
    print("LeWMTrainer GPU smoke (Phase E)")
    print("=" * 70)
    var ctx = DeviceContext()

    # max_grad_norm=1.0 exercises the GPU graph grad-clip path (Adam.step_graph
    # → clip_grads_graph_gpu, all 3 device passes run every step). The fix for
    # the CLS readout's mid-training gradient explosion; training must still
    # decrease with it on.
    var tr = Trainer.make(
        lam=Scalar[DT](0.09), lr=Scalar[DT](1e-3),
        max_grad_norm=Scalar[DT](1.0),
        weight_decay=Scalar[DT](1e-3),
        sigreg_resample=True, ctx=ctx,
    )

    # device IO + host staging for synthetic windows
    var pix_d = ctx.enqueue_create_buffer[DT](B * PIX)
    var act_d = ctx.enqueue_create_buffer[DT](B * ACTIN)
    var pix_h = ctx.enqueue_create_host_buffer[DT](B * PIX)
    var act_h = ctx.enqueue_create_host_buffer[DT](B * ACTIN)
    ctx.synchronize()

    # fixed synthetic batch (the smoke checks the device loop runs +
    # the loss falls on a memorisable batch — not generalisation)
    for k in range(B * PIX):
        pix_h.unsafe_ptr()[k] = _det(k + 1, 1.0)
    for k in range(B * ACTIN):
        act_h.unsafe_ptr()[k] = _det(k + 7, 1.0)
    ctx.enqueue_copy(pix_d, pix_h)
    ctx.enqueue_copy(act_d, act_h)
    ctx.synchronize()

    var pix_t = TileTensor(_p(pix_d), row_major[B, PIX]())
    var act_t = TileTensor(_p(act_d), row_major[B, ACTIN]())

    print("train ...")
    comptime STEPS = 120
    comptime WINDOW = 40
    var first_win: Scalar[DT] = 0.0
    var last_win: Scalar[DT] = 0.0
    var win_idx = 0
    tr.reset_loss_accum()
    for s in range(STEPS):
        _ = tr.train_step(pix_t, act_t)
        if (s + 1) % WINDOW == 0:
            var wl = tr.read_loss_accum()
            tr.reset_loss_accum()
            var probes = tr.collapse_probes()
            print("   step", s, " window_loss=", wl,
                  " var_min=", probes[0], " gram_off=", probes[1])
            assert_true(not (isnan(wl) or isinf(wl)), "window loss finite")
            if win_idx == 0:
                first_win = wl
            last_win = wl
            win_idx += 1
    print("   first_window=", first_win, " last_window=", last_win)
    assert_true(last_win < first_win, "loss must decrease over training")

    var probes = tr.collapse_probes()
    assert_true(not (isnan(probes[0]) or isinf(probes[0])), "var_min finite")
    assert_true(not (isnan(probes[1]) or isinf(probes[1])), "gram_off finite")

    # checkpoint round-trip on the SAME instance (SIGReg's projection is
    # pointer-seeded, stable within an instance — see CPU test note).
    # Resampling must be OFF here: with it on, every eval_loss forward
    # draws a fresh A → the SIGReg term (and thus the loss) differs run
    # to run even with identical params, breaking the exactness check.
    tr.graph.set_node_attr["sig", "resample"](Scalar[DT](0.0))
    print("checkpoint round-trip ...")
    var lA = tr.eval_loss(pix_t, act_t)
    var snap = tr.snapshot_all()
    tr.save_params(String("/tmp/lewm_ckpt_gpu.txt"))
    for _ in range(10):
        _ = tr.train_step(pix_t, act_t)
    var lA2 = tr.eval_loss(pix_t, act_t)
    tr.load_params(String("/tmp/lewm_ckpt_gpu.txt"))
    var lA3 = tr.eval_loss(pix_t, act_t)
    print("   lA=", lA, " perturbed=", lA2, " restored=", lA3)
    assert_true((lA2 - lA).__abs__() > Scalar[DT](1e-6),
                "training should perturb the eval loss (sanity)")
    assert_true((lA3 - lA).__abs__() < Scalar[DT](1e-4),
                "load_params must restore the saved model exactly")

    # `restore_all` (the AdaJEPA per-episode reset) after more training. A
    # restored weight that does not advance `param.version` leaves `Linear`'s
    # GPU padded-weight cache on the PERTURBED weight: the device values match
    # the snapshot bit for bit and the forward still differs. The forward
    # BEFORE the restore matters: it re-pads at the current version, so a
    # restore that does not bump is never re-padded. Without it the Adam step's
    # own bump hides the bug (plan -> reset is the AdaJEPA episode shape).
    for _ in range(10):
        _ = tr.train_step(pix_t, act_t)
    _ = tr.eval_loss(pix_t, act_t)
    tr.restore_all(snap)
    var lA4 = tr.eval_loss(pix_t, act_t)
    print("   restore_all after training=", lA4)
    assert_true((lA4 - lA).__abs__() < Scalar[DT](1e-4),
                "restore_all must restore the snapshot exactly")

    _ = tr^
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
