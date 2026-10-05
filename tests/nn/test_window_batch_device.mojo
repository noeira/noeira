"""`AutoregressiveTrainer[DEVICE_BATCH=True]`: the device-built training batch
is `make_batch`'s window batch, one-hot encoded.

Gates, each against the host corpus the trainer was given:
  - every row of `in_t` / `tgt_t` is the one-hot of corpus[s + t] /
    corpus[s + t + 1], with s the row's start as the device recorded it;
    elements exactly 0 or 1 (fp32 and bf16 activations);
  - the starts stay in [0, n_starts) and cover it (over 200 batches the
    extremes reach both ends), and the step word advances once per batch;
  - two trainers under the same `std.random.seed` draw the same batches,
    a different seed draws different ones.
Vacuity guard: the same comparison against targets taken at offset t (not
t + 1) must fail.

    pixi run -e apple mojo run -I . tests/nn/test_window_batch_device.mojo
"""

from std.random import seed, random_ui64
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.datasets import CharTokenizer, train_val_split
from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.primitives.linear import Linear
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.optimizer.adam import Adam
from noeira.nn.training.autoregressive_trainer import AutoregressiveTrainer
from noeira.nn.core.initializer import Normal


comptime VOCAB = 5
comptime SEQ = 16
comptime BATCH = 8
comptime IO = SEQ * VOCAB
comptime N_CHARS = 700
comptime N_STEPS = 200


def _text() -> String:
    seed(123)
    var alphabet = String("abcde")
    var text = String("")
    for _ in range(N_CHARS):
        var k = Int(random_ui64(0, 4))
        text += alphabet[byte=k : k + 1]
    return text^


def _trainer[
    NET: Module
](ctx: DeviceContext, rng_seed: Int) raises -> AutoregressiveTrainer[
    NET, Adam, VOCAB, SEQ, BATCH, target="gpu"
]:
    var text = _text()
    var tok = CharTokenizer(text)
    assert_true(tok.vocab_size == VOCAB, "tokenizer vocab")
    var split = train_val_split(tok.encode(text), 0.1)
    seed(rng_seed)
    var net = NET.make["gpu", INIT = Normal[0.0, 0.02]](Optional(ctx))
    var optim = Adam(lr=Scalar[DT](1e-3))
    return AutoregressiveTrainer[
        NET, Adam, VOCAB, SEQ, BATCH, target="gpu"
    ].make_from(
        net^, optim^, tok^, split^, ctx,
        Scalar[DT](1e-3), 2, 10, 0.1, Scalar[DT](1.0),
    )


def _mismatches[
    ADT: DType
](
    corpus: List[Int], starts: List[Int32], x: List[Scalar[ADT]],
    y: List[Scalar[DT]], target_shift: Int,
) -> Int:
    var bad = 0
    for b in range(BATCH):
        var s = Int(starts[b])
        for t in range(SEQ):
            for v in range(VOCAB):
                var i = b * IO + t * VOCAB + v
                var want_x = Float64(1.0) if corpus[s + t] == v else 0.0
                var want_y = (
                    Float64(1.0) if corpus[s + t + target_shift] == v else 0.0
                )
                if Float64(x[i]) != want_x:
                    bad += 1
                if Float64(y[i]) != want_y:
                    bad += 1
    return bad


def _gate[NET: Module](ctx: DeviceContext, label: String) raises:
    print("--", label)
    var tr = _trainer[NET](ctx, 7)
    var corpus = tr.train_ids.copy()
    var n_starts = len(corpus) - SEQ
    var lo = n_starts
    var hi = -1
    var first = List[Int32]()
    for k in range(N_STEPS):
        tr._device_batch()
        tr.starts_dev.download(ctx)
        tr.in_t.download(ctx)
        tr.tgt_t.download(ctx)
        for b in range(BATCH):
            var s = Int(tr.starts_dev.data[b])
            assert_true(s >= 0 and s < n_starts, "start out of range")
            lo = min(lo, s)
            hi = max(hi, s)
            if k == 0:
                first.append(tr.starts_dev.data[b])
        var bad = _mismatches[NET.ACT_DT](
            corpus, tr.starts_dev.data, tr.in_t.data, tr.tgt_t.data, 1
        )
        assert_true(bad == 0, String("one-hot mismatches: ") + String(bad))
        if k == 0:
            var shifted = _mismatches[NET.ACT_DT](
                corpus, tr.starts_dev.data, tr.in_t.data, tr.tgt_t.data, 0
            )
            print("   vacuity guard (targets at t, not t + 1):", shifted,
                  "mismatches")
            assert_true(shifted > 0, "the gate cannot see the target shift")
    tr.rng_dev.download(ctx)
    assert_true(
        tr.rng_dev.data[1] == UInt64(N_STEPS), "step word advanced per batch"
    )
    print("   starts span", lo, "..", hi, "of [0,", n_starts, ")")
    assert_true(lo < n_starts // 20 and hi > n_starts - n_starts // 20,
                "starts do not cover the corpus")

    # Same process seed -> same batches; another seed -> different ones.
    var same = _trainer[NET](ctx, 7)
    same._device_batch()
    same.starts_dev.download(ctx)
    var other = _trainer[NET](ctx, 8)
    other._device_batch()
    other.starts_dev.download(ctx)
    var n_same = 0
    var n_other = 0
    for b in range(BATCH):
        if same.starts_dev.data[b] == first[b]:
            n_same += 1
        if other.starts_dev.data[b] == first[b]:
            n_other += 1
    assert_true(n_same == BATCH, "same seed drew different windows")
    assert_true(n_other < BATCH, "another seed drew the same windows")
    print("   ok")


def main() raises:
    print("--- device window batches vs the host corpus ---")
    with DeviceContext() as ctx:
        _gate[Sequential[Linear[IO, IO]]](ctx, "fp32 activations")
        _gate[Sequential[Linear[IO, IO, DType.bfloat16]]](
            ctx, "bf16 activations"
        )
    print("ALL PASSED")
