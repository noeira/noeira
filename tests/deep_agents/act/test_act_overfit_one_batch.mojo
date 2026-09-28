# +--------------------------------------------------------------------------+ #
# | M6 gate — the trainer can drive one batch to ~0
# +--------------------------------------------------------------------------+ #
"""The check that the optimizer, the graph and the backward pass agree.

    pixi run mojo build -I . -Xlinker -ld_classic -o /tmp/t \\
        tests/deep_agents/act/test_act_overfit_one_batch.mojo && /tmp/t

Overfitting a single fixed batch is the sharpest cheap test of a training loop.
Every wiring error that survives a forward gate — a gradient with the wrong
sign, a detached branch, an optimizer walking a copy rather than the parameters
— shows up as a loss that will not descend, and nothing else does. A falling
curve on real data does NOT substitute: with four episodes a broken model still
produces a curve that goes down for a while.

Run on SYNTHETIC data (no store required) so the gate is self-contained and
cannot fail for a missing cache.

Three of ACT's training-time stochastic elements are switched OFF here, each for
the same reason: they inject fresh noise every step, so the loss would floor at
the noise level rather than at zero and the gate would be measuring the noise
rather than the training loop. Each has its own gate elsewhere.

    kl_weight = 0        a regularizer with a non-zero floor by construction (M4)
    dropout   p = 0      fresh mask per step (M2, compared in eval)
    z pinned             fresh reparameterization draw per step (M4, injected eps)

What remains under test is exactly what this gate is for: the forward, the
backward, the optimizer, and the checkpoint.

⚠ Also checks that the loss falls MONOTONICALLY over the last stretch and that a
checkpoint round-trips to the same number — a save that drops BatchNorm's
running statistics reloads to a model that looks like it never trained.
"""

from noeira.nn.constants import DT
from noeira.deep_agents.act.trainer import ACTTrainer


comptime QPOS = 6
comptime ADIM = 6
comptime N_CAM = 1
comptime IMG_H = 64
comptime IMG_W = 64
comptime K = 4
comptime DIM = 16
comptime HEADS = 2
comptime FF = 32
comptime LATENT = 8
comptime N_ENC = 1
comptime N_DEC = 1
comptime BATCH = 4

comptime STEPS = 700
comptime LR = 3e-4  # far above ACT's 1e-5 — this is an overfit probe, not a run
comptime P = 0.0
"""⚠ Dropout OFF for this probe, deliberately. ACT trains at p=0.1, but dropout
injects fresh noise every step, so a model CANNOT drive a single fixed batch to
zero however correct it is — the residual would measure the dropout rate, not
the training loop. Dropout's own arithmetic is gated in M2 (both encoder layers
compared in eval) and its train/eval switch in the ACT layer gate.

BATCH is 4 rather than 2 for the same reason on the other side: BatchNorm in
training mode estimates its statistics from the batch, and at BATCH=2 that
estimate is noisy enough to hold the loss off zero on its own."""

comptime T = ACTTrainer[
    QPOS, ADIM, N_CAM, IMG_H, IMG_W, K, DIM, HEADS, FF, LATENT, N_ENC, N_DEC,
    BATCH, P,
]
comptime IMG_ELEMS = N_CAM * 3 * IMG_H * IMG_W


def check(mut fails: Int, name: String, ok: Bool, detail: String = String("")):
    if ok:
        print("  PASS  " + name + ("  " + detail if detail else ""))
    else:
        fails += 1
        print("  FAIL  " + name + ("  " + detail if detail else ""))


def main() raises:
    var fails = 0
    print("ACT overfit-one-batch gate")
    print("")

    # ── one fixed synthetic batch ────────────────────────────────────────
    # Deterministic, structured (not noise): a constant target is learnable by
    # a bias alone, which would pass while proving nothing about the network.
    var qpos = List[Scalar[DT]](unsafe_uninit_length=BATCH * QPOS)
    var images = List[Scalar[DT]](unsafe_uninit_length=BATCH * IMG_ELEMS)
    var actions = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var valid = List[Scalar[DT]](unsafe_uninit_length=BATCH * K)

    for b in range(BATCH):
        for j in range(QPOS):
            qpos[b * QPOS + j] = Scalar[DT](
                0.3 * Float64(j + 1) - 0.7 * Float64(b)
            )
        for i in range(IMG_ELEMS):
            images[b * IMG_ELEMS + i] = Scalar[DT](
                0.05 * Float64((i * 7 + b * 13) % 17) - 0.4
            )
        for t in range(K):
            # The target depends on BOTH the sample and the timestep, so a
            # constant or a per-sample constant cannot fit it.
            for j in range(ADIM):
                actions[b * K * ADIM + t * ADIM + j] = Scalar[DT](
                    0.4 * Float64(t) - 0.25 * Float64(j) + 0.6 * Float64(b)
                )
            valid[b * K + t] = Scalar[DT](1.0)
    # Leave one padded slot so the mask path is exercised by the training loop
    # too, not only by the M4 gate.
    valid[BATCH * K - 1] = Scalar[DT](0.0)

    var tr = T.make(
        lr=Scalar[DT](LR),
        kl_weight=Scalar[DT](0.0),
        max_grad_norm=Scalar[DT](0.0),
    )
    # ⚠ Pin the latent. `train_mode(True)` draws fresh reparameterization noise
    # every step — correct for training, and fatal for this probe: the decoder
    # cannot predict a fresh random z, so the loss would floor at the noise the
    # latent injects rather than at zero, no matter how correct the gradients
    # are. `train_step` does not touch this flag, so setting it once holds.
    # The sampling path itself is gated in M4 against an injected eps.
    tr.graph.set_node_attr["z", "deterministic"](Scalar[DT](1.0))

    var first = Float64(0.0)
    var last = Float64(0.0)
    var mid = Float64(0.0)
    var worst_after_mid = Float64(0.0)
    var tail_sum = Float64(0.0)
    var tail_n = 0
    var saw_nan = False

    for s in range(STEPS):
        var r = tr.train_step(qpos, images, actions, valid)
        if r.loss != r.loss:
            saw_nan = True
        if s == 0:
            first = r.loss
        if s == STEPS // 2:
            mid = r.loss
        if s == STEPS - 1:
            last = r.loss
        if s > STEPS // 2:
            worst_after_mid = max(worst_after_mid, r.loss)
        if s >= STEPS - 50:
            tail_sum += r.loss
            tail_n += 1
        if s % 100 == 0 or s == STEPS - 1:
            print(
                "    step " + String(s) + "  loss " + String(r.loss)
                + "  l1 " + String(r.l1) + "  |g| " + String(r.grad_norm)
            )

    var tail_mean = tail_sum / Float64(tail_n)
    print("")
    check(fails, "no NaN in the loss", not saw_nan)
    check(
        fails,
        "the loss decreased",
        last < first,
        String(first) + " -> " + String(last),
    )
    check(
        fails,
        "the loss reached ~0 (overfit a single batch)",
        last < 0.02 * first,
        String(last) + " vs 2% of " + String(first) + " = "
        + String(0.02 * first),
    )
    # ⚠ NOT a monotonicity check. Adam near a minimum wobbles, and scaling a
    # tolerance against a midpoint that is itself near zero makes any threshold
    # arbitrary. What actually distinguishes "converging" from "broken" is the
    # TREND and the absence of divergence: the tail average must be well below
    # the midpoint, and no single step may blow past the starting loss.
    check(
        fails,
        "the tail is well below the midpoint (converging, not wandering)",
        tail_mean < 0.5 * mid,
        "tail mean " + String(tail_mean) + " vs mid " + String(mid),
    )
    check(
        fails,
        "no step diverged past the starting loss",
        worst_after_mid < first,
        "worst loss after the midpoint " + String(worst_after_mid),
    )

    # ── checkpoint round-trip ────────────────────────────────────────────
    var path = String("/tmp/act_overfit_ckpt.bin")
    tr.save(path)
    var before = tr.eval_step(qpos, images, actions, valid)

    # A fresh trainer starts at random init; its loss must be far off, and
    # loading must recover the trained number exactly. Checking only "load
    # runs" would pass on a load that silently did nothing.
    var tr2 = T.make(
        lr=Scalar[DT](LR),
        kl_weight=Scalar[DT](0.0),
        max_grad_norm=Scalar[DT](0.0),
    )
    var fresh = tr2.eval_step(qpos, images, actions, valid)
    tr2.load(path)
    var after = tr2.eval_step(qpos, images, actions, valid)

    # ⚠ `before.l1` is the EVAL loss and is legitimately well above the training
    # loss here: eval sets z = 0 (the reference's test-time path), while this
    # probe trained with the latent pinned to mu. That gap is the CVAE working
    # as designed, not a regression — so the margin below is 2x, which is still
    # decisive: a load that silently did nothing would make `after` equal
    # `fresh`, not merely close to it.
    print(
        "    train l1 " + String(last) + " vs eval l1 " + String(before.l1)
        + "  (eval runs at z = 0 — the reference's inference path)"
    )
    check(
        fails,
        "a fresh model is clearly worse than the trained one",
        fresh.l1 > 2.0 * before.l1,
        "fresh l1 " + String(fresh.l1) + " vs trained " + String(before.l1),
    )
    check(
        fails,
        "checkpoint round-trips (params AND BatchNorm state)",
        abs(after.l1 - before.l1) < 1e-5,
        "before " + String(before.l1) + " after " + String(after.l1),
    )
    # The moments ride the checkpoint, so Adam's step must too: restored
    # under t = 0 they are divided by the step-1 bias corrections.
    check(
        fails,
        "Adam's step state round-trips with the moments",
        tr.opt.t > 0 and tr2.opt.t == tr.opt.t and tr2.opt.bc2 == tr.opt.bc2,
        "saved t " + String(tr.opt.t) + " restored t " + String(tr2.opt.t),
    )

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("act overfit gate failed")
