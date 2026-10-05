"""VGG-style CNN on CIFAR-10 (GPU) — storage-surface (nn.storage) port.

Conv→BN→ReLU blocks with max-pooling, via the storage `Trainer.train_gpu`
whole-dataset API (resident upload + on-device per-epoch shuffle + BatchNorm
train/eval toggle). No data augmentation here (see the ResNet example for the
crop+flip recipe); bump N_EPOCHS and add an augmenter to push past ~75%.

Run (Apple): pixi run -e apple mojo run -I . examples/nn/conv2d/conv2d_cifar10_training_storage_gpu.mojo
Run (NVIDIA): pixi run -e nvidia mojo run -I . examples/nn/conv2d/conv2d_cifar10_training_storage_gpu.mojo
"""

from std.random import seed
from std.testing import assert_true
from std.sys import get_defined_int, is_defined
from max.gpu.host import DeviceContext

from noeira.nn.datasets import CIFAR10
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.models.conv import Conv2DBatchNormReLU
from noeira.nn.primitives.max_pool_2d import MaxPool2D
from noeira.nn.primitives.flatten import Flatten
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import ReLU
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.training.trainer import Trainer


# `-D NN_TRAIN_GRAPH`: the training step captured in a CUDA graph (NVIDIA;
# eager elsewhere). `-D NN_EPOCHS=N` shortens the run (step timing). Needs
# `mojo build -D ...`, then run the binary.
comptime TRAIN_GRAPH = is_defined["NN_TRAIN_GRAPH"]()

def main() raises:
    comptime IN_DIM = 3 * 32 * 32
    comptime NC = 10
    comptime BATCH = 100
    comptime N_EPOCHS = get_defined_int["NN_EPOCHS", 15]()
    comptime TARGET_ACC: Float64 = 0.65

    seed(42)
    print("loading CIFAR-10...")
    var ds = CIFAR10()
    var c = DeviceContext()

    comptime Net = Sequential[
        Conv2DBatchNormReLU[3, 32, 3, 1, 1, 32, 32],
        Conv2DBatchNormReLU[32, 32, 3, 1, 1, 32, 32],
        MaxPool2D[32, 2, 2, 0, 32, 32],  # → 32×16×16
        Conv2DBatchNormReLU[32, 64, 3, 1, 1, 16, 16],
        Conv2DBatchNormReLU[64, 64, 3, 1, 1, 16, 16],
        MaxPool2D[64, 2, 2, 0, 16, 16],  # → 64×8×8
        Conv2DBatchNormReLU[64, 128, 3, 1, 1, 8, 8],
        Conv2DBatchNormReLU[128, 128, 3, 1, 1, 8, 8],
        MaxPool2D[128, 2, 2, 0, 8, 8],  # → 128×4×4 = 2048
        Flatten[128 * 4 * 4],
        Linear[128 * 4 * 4, 128],
        ReLU[128],
        Linear[128, NC],
    ]
    print("initializing network (GPU)...")
    var trainer = Trainer[
        Net, NC, IN_DIM, BATCH, "gpu", USE_TRAIN_CUDA_GRAPH=TRAIN_GRAPH
    ].make[Kaiming](
        Optional(c), lr=1e-3
    )

    var train_y = List[Scalar[DT]](length=CIFAR10.N_TRAIN * NC, fill=0.0)
    for i in range(CIFAR10.N_TRAIN):
        train_y[i * NC + Int(ds.train_labels[i])] = 1.0

    var result = trainer.train_gpu[CIFAR10.N_TRAIN, CIFAR10.N_TEST](
        ds.train_images,
        train_y,
        ds.test_images,
        ds.test_labels,
        Optional(c),
        epochs=N_EPOCHS,
        shuffle=True,
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
