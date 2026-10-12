"""MLP on MNIST (GPU) — storage-surface (nn.storage) port.

The GPU twin of mlp_mnist_training_storage_cpu.mojo. Same storage Trainer /
Sequential[LinearReLU, LinearReLU, Linear] / CrossEntropyLoss / Adam / Kaiming,
with target="gpu". The Trainer uploads the dataset to device ONCE and slices each
batch as a zero-copy `create_sub_buffer` view (no per-batch H2D — matches legacy
data movement). The caller owns the DeviceContext and reuses it for make/train/eval.

Run (Apple):  pixi run -e apple mojo run -I . examples/nn/mlp/mlp_mnist_training_storage_gpu.mojo
Run (NVIDIA): pixi run -e nvidia mojo run -I . examples/nn/mlp/mlp_mnist_training_storage_gpu.mojo
"""

from std.random import seed
from std.testing import assert_true
from std.time import perf_counter_ns
from std.sys import get_defined_int, is_defined
from max.gpu.host import DeviceContext

from noeira.nn.datasets import MNIST
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.linear_relu import LinearReLU
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.training.trainer import Trainer


# `-D NN_TRAIN_GRAPH`: the training step captured in a CUDA graph (NVIDIA;
# eager elsewhere). `-D NN_EPOCHS=N` shortens the run (step timing). Needs
# `mojo build -D ...`, then run the binary.
comptime TRAIN_GRAPH = is_defined["NN_TRAIN_GRAPH"]()

def main() raises:
    comptime IN_DIM = 784
    comptime H1 = 256
    comptime H2 = 128
    comptime NC = 10
    comptime BATCH = 100
    comptime N_EPOCHS = get_defined_int["NN_EPOCHS", 5]()
    comptime TARGET_ACC: Float64 = 0.97

    seed(42)
    print("loading MNIST...")
    var ds = MNIST()
    var c = DeviceContext()

    comptime Net = Sequential[
        LinearReLU[IN_DIM, H1],
        LinearReLU[H1, H2],
        Linear[H2, NC],
    ]
    print("initializing network (GPU)...")
    var trainer = Trainer[
        Net, NC, IN_DIM, BATCH, "gpu", USE_TRAIN_CUDA_GRAPH=TRAIN_GRAPH
    ].make[Kaiming](
        Optional(c), lr=1e-3
    )

    var train_y = List[Scalar[DT]](length=MNIST.N_TRAIN * NC, fill=0.0)
    for i in range(MNIST.N_TRAIN):
        train_y[i * NC + Int(ds.train_labels[i])] = 1.0

    # One-call whole-dataset run with on-device per-epoch shuffle.
    var t0 = perf_counter_ns()
    var result = trainer.train_gpu[MNIST.N_TRAIN, MNIST.N_TEST](
        ds.train_images,
        train_y,
        ds.test_images,
        ds.test_labels,
        Optional(c),
        epochs=N_EPOCHS,
        shuffle=True,
    )
    var total_s = Float64(perf_counter_ns() - t0) / 1e9

    var best_acc: Float64 = 0.0
    for a in result.epoch_test_top1:
        if a > best_acc:
            best_acc = a

    print("\nbest test accuracy: " + String(best_acc * 100.0) + "%")
    print("total wall time: " + String(total_s) + "s")
    assert_true(
        best_acc >= TARGET_ACC,
        "Expected best >= " + String(TARGET_ACC * 100.0) + "%, got "
        + String(best_acc * 100.0) + "%",
    )
    print("DONE")
