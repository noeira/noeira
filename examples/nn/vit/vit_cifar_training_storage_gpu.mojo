"""Vision Transformer (ViT) on CIFAR-10 (GPU) — storage-surface (nn.storage) port.

Patch-embeds 32×32 images, runs a transformer encoder, and classifies from the
CLS token. Trained through the storage `Trainer.train_gpu` whole-run with all
Tier-B features: on-device shuffle + per-epoch CIFAR crop+flip augmentation +
`WarmupCosineSchedule` LR, with AdamW decoupled weight decay.

Default config is sized for NVIDIA (≥70% target). On Apple it OOMs — shrink to
PATCH=8/EMBED=64/HEADS=4/LAYERS=3/FF_MULT=2/BATCH=64/EPOCHS=8 for a dev run.

Run (NVIDIA): pixi run -e nvidia mojo run -I . examples/nn/vit/vit_cifar_training_storage_gpu.mojo
"""

from std.random import seed
from std.testing import assert_true
from std.time import perf_counter_ns
from std.sys import get_defined_int, is_defined
from max.gpu.host import DeviceContext

from noeira.nn.datasets import CIFAR10
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.models.vit import ViT
from noeira.nn.optimizer.adam import AdamW
from noeira.nn.training.trainer import Trainer
from noeira.nn.training.augmenter import CIFAR10CropFlipAugmenter
from noeira.nn.optimizer.lr_scheduler import WarmupCosineSchedule


comptime IN_CHANNELS = 3
comptime IMG_H = 32
comptime IMG_W = 32
comptime PATCH = 4  # 8×8 = 64 patches
comptime N_PATCHES = (IMG_H // PATCH) * (IMG_W // PATCH)
comptime EMBED = 192
comptime HEADS = 6  # head_dim = 32
comptime LAYERS = 6
comptime FF_MULT = 4
comptime NC = 10
comptime IN_DIM = IN_CHANNELS * IMG_H * IMG_W

comptime BATCH = 128
comptime EPOCHS = get_defined_int["NN_EPOCHS", 100]()
comptime WARMUP_EPOCHS = 5
# `-D NN_TRAIN_GRAPH`: the training step captured in a CUDA graph (NVIDIA;
# eager elsewhere). `-D NN_EPOCHS=N` shortens the run (step timing). Needs
# `mojo build -D ...`, then run the binary.
comptime TRAIN_GRAPH = is_defined["NN_TRAIN_GRAPH"]()
comptime BASE_LR: Scalar[DT] = 3e-4
comptime WD: Scalar[DT] = 0.05

# Drop trailing partial batch (train_gpu asserts BATCH-divisibility).
comptime N_TRAIN = (CIFAR10.N_TRAIN // BATCH) * BATCH
comptime N_TEST = (CIFAR10.N_TEST // BATCH) * BATCH

comptime VIT_MODEL = ViT[
    IN_CHANNELS, IMG_H, IMG_W, PATCH, EMBED, HEADS, LAYERS, N_PATCHES, NC,
    FF_MULT,
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("ViT CIFAR-10 training — nn.storage GPU")
    print("=" * 70)

    print("\n[data] loading CIFAR-10...")
    var ds = CIFAR10()
    var c = DeviceContext()

    print("[init] building storage ViT on GPU...")
    var trainer = Trainer[
        VIT_MODEL, NC, IN_DIM, BATCH, "gpu", OPT=AdamW,
        USE_TRAIN_CUDA_GRAPH=TRAIN_GRAPH,
    ].make[Kaiming](Optional(c), lr=BASE_LR)
    # AdamW = storage Adam with decoupled weight decay; scalars read fresh each
    # step, so setting them post-make (after arena adopt) is fine.
    trainer.opt.wd = WD
    trainer.opt.beta2 = Scalar[DT](0.999)

    var train_y = List[Scalar[DT]](length=CIFAR10.N_TRAIN * NC, fill=0.0)
    for i in range(CIFAR10.N_TRAIN):
        train_y[i * NC + Int(ds.train_labels[i])] = 1.0

    print("\n── Training ──")
    var t0 = perf_counter_ns()
    var result = trainer.train_gpu[
        N_TRAIN,
        N_TEST,
        CIFAR10CropFlipAugmenter,
        WarmupCosineSchedule[WARMUP_EPOCHS, 0.1],
    ](
        ds.train_images,
        train_y,
        ds.test_images,
        ds.test_labels,
        Optional(c),
        epochs=EPOCHS,
        shuffle=True,
        rng_seed=UInt64(42),
        aug_seed=UInt64(1000),
    )
    print("  training time: " + String(Float64(perf_counter_ns() - t0) / 1e9) + " s")

    var best: Float64 = 0.0
    for a in result.epoch_test_top1:
        if a > best:
            best = a
    var n = len(result.epoch_test_top1)
    print("\n── Final ──")
    print("  final top-1: " + String(result.epoch_test_top1[n - 1] * 100.0) + "%")
    print("  best top-1:  " + String(best * 100.0) + "%")
    assert_true(best >= 0.70, "Expected best >= 70%, got " + String(best * 100.0))
    print("DONE")
