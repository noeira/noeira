"""`Trainer.train_gpu` with `USE_TRAIN_CUDA_GRAPH`: the captured step (batch
gather at a device counter + compute) reproduces the eager run bit for bit,
with on-device shuffle, crop+flip augmentation and an LR schedule, on a
BatchNorm CNN.

Gates, each eager vs capture on the same seeds:
  - every epoch's train loss and test top-1 are equal;
  - the checkpoints (weights + BN running statistics) are byte-identical;
  - also for the contiguous sweep (no shuffle, no augmentation), which the
    capture path now runs as the identity permutation.
Non-vacuity: the weights moved off their initialisation; the captured run
with another shuffle seed, or another augmentation seed, ends elsewhere (the
graph reads the epoch's permutation and augmented set, not a frozen batch).

On Metal `maybe_capture_replay` runs the step eagerly, so this compares the
device-counter gather against the host-offset gather; on NVIDIA it also
replays a real graph.

    pixi run -e apple mojo run -I . tests/nn/test_trainer_capture_shuffle_aug.mojo
"""

from std.random import seed
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.checkpoint import save_params
from noeira.nn.core.initializer import Kaiming
from noeira.nn.models.conv import Conv2DBatchNormReLU
from noeira.nn.primitives.max_pool_2d import MaxPool2D
from noeira.nn.primitives.flatten import Flatten
from noeira.nn.primitives.linear import Linear
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.training.trainer import Trainer, TrainResult
from noeira.nn.training.augmenter import (
    Augmenter, IdentityAugmenter, CIFAR10CropFlipAugmenter,
)
from noeira.nn.optimizer.lr_scheduler import (
    Scheduler, ConstantSchedule, WarmupCosineSchedule,
)


comptime IN = 3 * 32 * 32
comptime NC = 10
comptime BATCH = 40
comptime N_TRAIN = 200
comptime N_TEST = 40
comptime EPOCHS = 3
comptime Net = Sequential[
    Conv2DBatchNormReLU[3, 4, 3, 1, 1, 32, 32],
    MaxPool2D[4, 2, 2, 0, 32, 32],
    Flatten[4 * 16 * 16],
    Linear[4 * 16 * 16, NC],
]


struct Data(Movable):
    var x: List[Scalar[DT]]
    var y: List[Scalar[DT]]
    var tx: List[Scalar[DT]]
    var tl: List[Int32]

    def __init__(out self):
        seed(5)
        self.x = List[Scalar[DT]](length=N_TRAIN * IN, fill=0.0)
        self.y = List[Scalar[DT]](length=N_TRAIN * NC, fill=0.0)
        self.tx = List[Scalar[DT]](length=N_TEST * IN, fill=0.0)
        self.tl = List[Int32](length=N_TEST, fill=0)
        # A class-dependent central patch plus a per-image ramp, so rows differ
        # (a shuffle that misreads the permutation changes the batches).
        for i in range(N_TRAIN):
            var cls = i % NC
            for k in range(IN):
                var row = (k % 1024) // 32
                var col = k % 32
                var v = Float64(i % 7) * 0.01 + Float64(col) * 0.002
                if row >= 8 and row < 24 and col >= 8 and col < 24:
                    v += Float64(cls + 1) * 0.1
                self.x[i * IN + k] = Scalar[DT](v)
            self.y[i * NC + cls] = 1.0
        for i in range(N_TEST):
            var cls = (i * 3) % NC
            for k in range(IN):
                var row = (k % 1024) // 32
                var col = k % 32
                if row >= 8 and row < 24 and col >= 8 and col < 24:
                    self.tx[i * IN + k] = Scalar[DT](Float64(cls + 1) * 0.1)
            self.tl[i] = Int32(cls)


def _read(path: String) raises -> List[UInt8]:
    with open(path, "r") as f:
        return f.read_bytes()


def _same_bytes(a: String, b: String) raises -> Bool:
    var x = _read(a)
    var y = _read(b)
    if len(x) != len(y):
        return False
    for i in range(len(x)):
        if x[i] != y[i]:
            return False
    return True


def _run[
    CAP: Bool, AUG: Augmenter, SCHED: Scheduler
](
    ctx: DeviceContext, ref d: Data, shuffle: Bool, rng_seed: UInt64,
    aug_seed: UInt64, tag: String,
) raises -> TrainResult:
    seed(42)
    var tr = Trainer[
        Net, NC, IN, BATCH, "gpu", USE_TRAIN_CUDA_GRAPH=CAP
    ].make[Kaiming](Optional(ctx), lr=3e-3)
    save_params["gpu"](tr.model, "/tmp/trainer_cap_init_" + tag + ".ckpt", ctx)
    var res = tr.train_gpu[N_TRAIN, N_TEST, AUG, SCHED](
        d.x, d.y, d.tx, d.tl, Optional(ctx), epochs=EPOCHS, shuffle=shuffle,
        print_progress=False, rng_seed=rng_seed, aug_seed=aug_seed,
    )
    save_params["gpu"](tr.model, "/tmp/trainer_cap_" + tag + ".ckpt", ctx)
    for e in range(EPOCHS):
        print("  ", tag, "| epoch", e, "| loss", res.epoch_train_loss[e],
              "| top-1", res.epoch_test_top1[e])
    return res^


def _equal(a: TrainResult, b: TrainResult, what: String) raises:
    for e in range(EPOCHS):
        assert_true(
            a.epoch_train_loss[e] == b.epoch_train_loss[e],
            what + ": train loss differs at epoch " + String(e),
        )
        assert_true(
            a.epoch_test_top1[e] == b.epoch_test_top1[e],
            what + ": top-1 differs at epoch " + String(e),
        )


def main() raises:
    print("--- Trainer.train_gpu: eager vs captured step ---")
    var d = Data()
    with DeviceContext() as ctx:
        comptime Aug = CIFAR10CropFlipAugmenter
        comptime Sch = WarmupCosineSchedule[1, 0.1]
        var e = _run[False, Aug, Sch](ctx, d, True, 42, 1000, "eager_sa")
        var g = _run[True, Aug, Sch](ctx, d, True, 42, 1000, "cap_sa")
        _equal(e, g, "shuffle+aug")
        assert_true(
            _same_bytes("/tmp/trainer_cap_eager_sa.ckpt", "/tmp/trainer_cap_cap_sa.ckpt"),
            "shuffle+aug: checkpoints differ",
        )
        assert_true(
            not _same_bytes("/tmp/trainer_cap_init_cap_sa.ckpt", "/tmp/trainer_cap_cap_sa.ckpt"),
            "the weights never moved",
        )
        print("  shuffle+aug: identical")

        var gs = _run[True, Aug, Sch](ctx, d, True, 43, 1000, "cap_seed")
        assert_true(
            not _same_bytes("/tmp/trainer_cap_cap_sa.ckpt", "/tmp/trainer_cap_cap_seed.ckpt"),
            "another shuffle seed ended at the same weights",
        )
        var ga = _run[True, Aug, Sch](ctx, d, True, 42, 1001, "cap_aug")
        assert_true(
            not _same_bytes("/tmp/trainer_cap_cap_sa.ckpt", "/tmp/trainer_cap_cap_aug.ckpt"),
            "another augmentation seed ended at the same weights",
        )
        _ = gs^
        _ = ga^

        comptime Id = IdentityAugmenter
        comptime Cst = ConstantSchedule
        var ec = _run[False, Id, Cst](ctx, d, False, 42, 1000, "eager_c")
        var gc = _run[True, Id, Cst](ctx, d, False, 42, 1000, "cap_c")
        _equal(ec, gc, "contiguous")
        assert_true(
            _same_bytes("/tmp/trainer_cap_eager_c.ckpt", "/tmp/trainer_cap_cap_c.ckpt"),
            "contiguous: checkpoints differ",
        )
        print("  contiguous: identical")
    print("ALL PASSED")
