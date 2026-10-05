"""ResNet-20 on CIFAR-10 (GPU) — storage-surface (nn.storage) port.

ResNet-20 (He et al. 2016, CIFAR variant; 6n+2 with n=3): a 3×3 BN stem, 3
stages of 3 residual blocks (channels 16→32→64, stages 2–3 downsampling), then
global average pool + linear head. Built from the storage residual composites
(`ResBlockConv2DBN` / `ResBlockDownsampleBN`).

Uses the storage `Trainer.train_gpu` whole-run with all Tier-B features: on-device
shuffle + per-epoch CIFAR crop+flip augmentation (`CIFAR10CropFlipAugmenter`) +
BatchNorm train/eval toggle + `WarmupCosineSchedule` LR (5-epoch warmup then
cosine decay to 1% of base). With Adam + augmentation + cosine LR + 50 epochs
this clears ~80%+. ResNet-20 is deep — expect a long compile.

Run (Apple): pixi run -e apple mojo run -I . examples/nn/resnet/resnet20_cifar10_training_storage_gpu.mojo
Run (NVIDIA): pixi run -e nvidia mojo run -I . examples/nn/resnet/resnet20_cifar10_training_storage_gpu.mojo
"""

from std.random import seed
from std.testing import assert_true
from std.sys import is_defined
from max.gpu.host import DeviceContext

from noeira.nn.datasets import CIFAR10
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.models.conv import Conv2DBatchNormReLU
from noeira.nn.models.resnet import (
    ResBlockConv2DBN, ResBlockDownsampleBN,
)
from noeira.nn.primitives.avg_pool_2d import AvgPool2D
from noeira.nn.primitives.flatten import Flatten
from noeira.nn.primitives.linear import Linear
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.combinators.repeat import Repeat
from noeira.nn.training.trainer import Trainer
from noeira.nn.training.augmenter import CIFAR10CropFlipAugmenter
from noeira.nn.optimizer.lr_scheduler import WarmupCosineSchedule


# `-D NN_TRAIN_GRAPH`: the training step captured in a CUDA graph (NVIDIA;
# eager elsewhere). Needs `mojo build -D ...`, then run the binary.
comptime TRAIN_GRAPH = is_defined["NN_TRAIN_GRAPH"]()

def main() raises:
    comptime IN_DIM = 3 * 32 * 32
    comptime NC = 10
    comptime BATCH = 100
    comptime N_EPOCHS = 50
    comptime TARGET_ACC: Float64 = 0.80

    seed(42)
    print("loading CIFAR-10...")
    var ds = CIFAR10()
    var c = DeviceContext()

    comptime Net = Sequential[
        Conv2DBatchNormReLU[3, 16, 3, 1, 1, 32, 32],  # stem → 16×32×32
        # Stage 1: 3 identity blocks @ 16ch, 32×32
        Repeat[3, ResBlockConv2DBN[16, 3, 1, 32, 32], shared=False],
        # Stage 2: downsample 16→32 (32×32→16×16) + 2 identity blocks
        ResBlockDownsampleBN[16, 32, 3, 1, 32, 32],
        Repeat[2, ResBlockConv2DBN[32, 3, 1, 16, 16], shared=False],
        # Stage 3: downsample 32→64 (16×16→8×8) + 2 identity blocks
        ResBlockDownsampleBN[32, 64, 3, 1, 16, 16],
        Repeat[2, ResBlockConv2DBN[64, 3, 1, 8, 8], shared=False],
        # Head: global avg pool → 64, linear
        AvgPool2D[64, 8, 8, 0, 8, 8],
        Flatten[64],
        Linear[64, NC],
    ]

    print("initializing ResNet-20 on GPU (this compile is long)...")
    var trainer = Trainer[
        Net, NC, IN_DIM, BATCH, "gpu", USE_TRAIN_CUDA_GRAPH=TRAIN_GRAPH
    ].make[Kaiming](
        Optional(c), lr=1e-3
    )

    var train_y = List[Scalar[DT]](length=CIFAR10.N_TRAIN * NC, fill=0.0)
    for i in range(CIFAR10.N_TRAIN):
        train_y[i * NC + Int(ds.train_labels[i])] = 1.0

    # The trainer handles shuffling, per-epoch CIFAR crop+flip augmentation, the
    # BatchNorm train/eval toggle, the LR schedule (5-epoch warmup then cosine
    # decay to 1% of base), and per-epoch top-1 eval internally.
    var result = trainer.train_gpu[
        CIFAR10.N_TRAIN,
        CIFAR10.N_TEST,
        CIFAR10CropFlipAugmenter,
        WarmupCosineSchedule[5, 0.01],
    ](
        ds.train_images,
        train_y,
        ds.test_images,
        ds.test_labels,
        Optional(c),
        epochs=N_EPOCHS,
        shuffle=True,
        rng_seed=UInt64(42),
        aug_seed=UInt64(1000),
    )

    var best_acc: Float64 = 0.0
    for a in result.epoch_test_top1:
        if a > best_acc:
            best_acc = a
    print("\nbest test accuracy: " + String(best_acc * 100.0) + "%")
    assert_true(
        best_acc >= TARGET_ACC,
        "Expected best >= " + String(TARGET_ACC * 100.0) + "%, got "
        + String(best_acc * 100.0) + "%",
    )
    print("DONE")
