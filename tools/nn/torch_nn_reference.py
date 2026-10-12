#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | The nn examples, in PyTorch, for the throughput comparison
# +--------------------------------------------------------------------------+ #
"""PyTorch twins of the five `examples/nn` training runs: the same model,
recipe, batch size and data files, so a wall-clock comparison against our
examples reads the framework, not the experiment.

    pixi run -e act-ref python tools/nn/torch_nn_reference.py mlp
    pixi run -e act-ref python tools/nn/torch_nn_reference.py resnet20 --mode compile
    pixi run -e act-ref python tools/nn/torch_nn_reference.py gpt --mode cudagraph

On the Linux box run it WITHOUT the project's LD_PRELOAD (the CUDA interceptor
stalls torch's CUDA build — see `dump-lewm-ref` in pixi.toml):

    env -u LD_PRELOAD .pixi/envs/act-ref/bin/python tools/nn/torch_nn_reference.py gpt

| model    | ours                                              | recipe |
|----------|---------------------------------------------------|--------|
| mlp      | mlp/mlp_mnist_training_storage_gpu.mojo           | 784-256-128-10, Adam 1e-3, batch 100, 5 epochs, shuffle |
| cnn      | conv2d/conv2d_cifar10_training_storage_gpu.mojo   | 6 conv-BN-ReLU + 3 pools + 2 Linear, Adam 1e-3, batch 100, 15 epochs, shuffle |
| resnet20 | resnet/resnet20_cifar10_training_storage_gpu.mojo | ResNet-20, Adam 1e-3, batch 100, 50 epochs, crop+flip, warmup 5 + cosine to 0.01 |
| vit      | vit/vit_cifar_training_storage_gpu.mojo           | patch 4, dim 192, 6 heads, 6 layers, AdamW 3e-4 wd 0.05, batch 128, 100 epochs, crop+flip, warmup 5 + cosine to 0.1 |
| gpt      | transformer/gpt_tinyshakespeare_training_gpu.mojo | char GPT 6x384, seq 256, batch 64, dropout 0.2, tied head, AdamW 1e-3 (0.9, 0.99) wd 0.1, clip 1, 5000 iters, warmup 100 + cosine to 0.1 |

## What is held equal, and what is not

- **The data lives on the GPU on both sides.** Ours uploads each dataset once
  and shuffles / augments / samples on the device; so does this script
  (`randperm`, a whole-set crop+flip per epoch, device window sampling). A
  `DataLoader` with CPU workers would compare our device pipeline against a
  host one — a weaker baseline, not the framework.
- **Precision.** Our CUDA GEMMs run TF32 (MAX's multistage matmul cannot turn
  it off for float32 outside SM100), so `--tf32 on` is the default here.
  `--tf32 off` gives strict float32 for reference.
- **Three PyTorch columns** (`--mode`): `eager`; `compile` (`torch.compile`);
  `cudagraph` (`torch.compile(mode="reduce-overhead")`, CUDA graphs like our
  `-D NN_TRAIN_GRAPH`). Adam / AdamW run `fused=True` in every mode (CUDA).
- **Same architecture and recipe**, incl. the GELU flavour (tanh, as ours),
  BatchNorm momentum 0.1 / eps 1e-5, conv biases before BN, He-uniform init
  for the classifiers, N(0, 0.02) for the GPT. Not bit-identical: the RNG
  streams (init, shuffle, augmentation, dropout) differ, so accuracies match
  in distribution, not per epoch.

Timing matches the examples' own lines: per epoch, train seconds (the epoch's
batches, then a sync) and eval seconds separately; the GPT reports the whole
`fit` (training plus its periodic evaluation) and, with `--bench-steps`, the
steady-state step time alone. A final `RESULT {json}` line carries the numbers.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = os.path.expanduser(os.environ.get("NOEIRA_CACHE", "~/.cache/noeira"))
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------- data ------ #
def load_mnist(dev):
    def images(name):
        raw = np.fromfile(os.path.join(CACHE, "mnist", name), dtype=np.uint8)
        return torch.from_numpy(raw[16:].reshape(-1, 784).astype(np.float32) / 255.0)

    def labels(name):
        raw = np.fromfile(os.path.join(CACHE, "mnist", name), dtype=np.uint8)
        return torch.from_numpy(raw[8:].astype(np.int64))

    return (images("train-images-idx3-ubyte").to(dev),
            labels("train-labels-idx1-ubyte").to(dev),
            images("t10k-images-idx3-ubyte").to(dev),
            labels("t10k-labels-idx1-ubyte").to(dev))


def load_cifar10(dev):
    root = os.path.join(CACHE, "cifar10", "cifar-10-batches-bin")

    def read(names):
        rows = np.concatenate([
            np.fromfile(os.path.join(root, n), dtype=np.uint8).reshape(-1, 3073)
            for n in names
        ])
        y = torch.from_numpy(rows[:, 0].astype(np.int64))
        x = torch.from_numpy(rows[:, 1:].astype(np.float32) / 255.0).view(-1, 3, 32, 32)
        mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
        return ((x - mean) / std).to(dev), y.to(dev)

    tx, ty = read([f"data_batch_{i}.bin" for i in range(1, 6)])
    vx, vy = read(["test_batch.bin"])
    return tx, ty, vx, vy


def crop_flip(x, gen):
    """Pad-4 random crop + horizontal flip of the WHOLE set, per sample (as our
    `CIFAR10CropFlipAugmenter`: rewritten from the raw set once per epoch).
    Padding is 0 on the normalised image, as ours."""
    n = x.shape[0]
    pad = F.pad(x, (4, 4, 4, 4))
    dx = torch.randint(0, 9, (n,), device=x.device, generator=gen)
    dy = torch.randint(0, 9, (n,), device=x.device, generator=gen)
    flip = torch.rand(n, device=x.device, generator=gen) < 0.5
    ar = torch.arange(32, device=x.device)
    cols = dx[:, None] + ar[None, :]                       # (n, 32)
    cols = torch.where(flip[:, None], dx[:, None] + 31 - ar[None, :], cols)
    rows = dy[:, None] + ar[None, :]
    bi = torch.arange(n, device=x.device)[:, None, None, None]
    ci = torch.arange(3, device=x.device)[None, :, None, None]
    return pad[bi, ci, rows[:, None, :, None], cols[:, None, None, :]]


# -------------------------------------------------------------- models ------ #
def conv_bn_relu(i, o, stride=1):
    return [nn.Conv2d(i, o, 3, stride, 1), nn.BatchNorm2d(o), nn.ReLU()]


class Block(nn.Module):
    def __init__(self, i, o, stride):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(i, o, 3, stride, 1), nn.BatchNorm2d(o), nn.ReLU(),
            nn.Conv2d(o, o, 3, 1, 1), nn.BatchNorm2d(o))
        self.skip = (nn.Identity() if stride == 1 and i == o else
                     nn.Sequential(nn.Conv2d(i, o, 1, stride, 0), nn.BatchNorm2d(o)))

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


def make_mlp():
    return nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128),
                         nn.ReLU(), nn.Linear(128, 10))


def make_cnn():
    return nn.Sequential(
        *conv_bn_relu(3, 32), *conv_bn_relu(32, 32), nn.MaxPool2d(2),
        *conv_bn_relu(32, 64), *conv_bn_relu(64, 64), nn.MaxPool2d(2),
        *conv_bn_relu(64, 128), *conv_bn_relu(128, 128), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(2048, 128), nn.ReLU(), nn.Linear(128, 10))


def make_resnet20():
    layers = [*conv_bn_relu(3, 16)]
    layers += [Block(16, 16, 1) for _ in range(3)]
    layers += [Block(16, 32, 2)] + [Block(32, 32, 1) for _ in range(2)]
    layers += [Block(32, 64, 2)] + [Block(64, 64, 1) for _ in range(2)]
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)]
    return nn.Sequential(*layers)


class Attn(nn.Module):
    def __init__(self, d, h, causal, p_drop=0.0):
        super().__init__()
        self.h, self.causal = h, causal
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.drop(self.proj(y.transpose(1, 2).reshape(b, t, d)))


class TBlock(nn.Module):
    """Pre-LN block, tanh GELU (as our `TransformerBlock` / `TransformerBlockDrop`)."""

    def __init__(self, d, h, ff, causal, p_drop=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, h, causal, p_drop)
        self.fc1, self.fc2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh")))


class ViT(nn.Module):
    """Patch conv -> learned position bias -> 6 blocks -> LN -> token mean -> head."""

    def __init__(self, d=192, h=6, layers=6, patch=4, nc=10):
        super().__init__()
        n = (32 // patch) ** 2
        self.patch = nn.Conv2d(3, d, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, n, d))
        self.blocks = nn.Sequential(*[TBlock(d, h, 4 * d, False) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, nc)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        return self.head(self.ln(self.blocks(x)).mean(1))


class GPT(nn.Module):
    def __init__(self, vocab=65, seq=256, d=384, h=6, layers=6, p_drop=0.2):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, seq, d))
        self.drop = nn.Dropout(p_drop)
        self.blocks = nn.Sequential(*[TBlock(d, h, 4 * d, True, p_drop) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight          # tied, as `GPTDropTied`
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, 0.0, 0.02)
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                with torch.no_grad():
                    p.mul_(1.0 / math.sqrt(2 * layers))  # gpt_scale_residual_proj

    def forward(self, idx):
        return self.head(self.ln(self.blocks(self.drop(self.tok(idx) + self.pos))))


def he_uniform_(model):
    """Our `Kaiming`: U(-b, b), b = sqrt(6 / fan_in), on every conv / linear
    weight; biases zero."""
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def warmup_cosine(epoch, total, warmup, min_scale):
    """Our `WarmupCosineSchedule[warmup, min_scale].lr_scale_at`."""
    if epoch < warmup:
        return (epoch + 1) / warmup
    prog = min(1.0, (epoch - warmup) / max(1, total - warmup))
    return min_scale + (1.0 - min_scale) * 0.5 * (1.0 + math.cos(math.pi * prog))


# ------------------------------------------------------------- drivers ------ #
def wrap(model, mode):
    if mode == "compile":
        return torch.compile(model)
    if mode == "cudagraph":
        return torch.compile(model, mode="reduce-overhead")
    return model


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


CLASSIFIERS = {
    "mlp": dict(make=make_mlp, data=load_mnist, batch=100, epochs=5, lr=1e-3,
                wd=0.0, aug=False, sched=None, flat=True),
    "cnn": dict(make=make_cnn, data=load_cifar10, batch=100, epochs=15, lr=1e-3,
                wd=0.0, aug=False, sched=None, flat=False),
    "resnet20": dict(make=make_resnet20, data=load_cifar10, batch=100, epochs=50,
                     lr=1e-3, wd=0.0, aug=True, sched=(5, 0.01), flat=False),
    "vit": dict(make=ViT, data=load_cifar10, batch=128, epochs=100, lr=3e-4,
                wd=0.05, aug=True, sched=(5, 0.1), flat=False),
}


def run_classifier(name, args, dev):
    cfg = CLASSIFIERS[name]
    torch.manual_seed(args.seed)
    tx, ty, vx, vy = cfg["data"](dev)
    b = cfg["batch"]
    n_train = (tx.shape[0] // b) * b
    n_test = (vx.shape[0] // b) * b
    epochs = args.epochs or cfg["epochs"]
    net = cfg["make"]().to(dev)
    if name != "vit":
        he_uniform_(net)
    model = wrap(net, args.mode)
    fused = dev.type == "cuda"
    opt = (torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"], fused=fused)
           if cfg["wd"] > 0 else
           torch.optim.Adam(net.parameters(), lr=cfg["lr"], fused=fused))
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    def step(xb, yb):
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    out = dict(model=name, mode=args.mode, tf32=args.tf32, batch=b,
               epoch_train_s=[], epoch_eval_s=[], epoch_top1=[], epoch_loss=[])
    print(f"[{name}] mode={args.mode} tf32={args.tf32} batch={b} epochs={epochs} "
          f"torch={torch.__version__} device={torch.cuda.get_device_name() if dev.type == 'cuda' else dev}")
    for epoch in range(epochs):
        sync()
        t0 = time.perf_counter()
        if cfg["sched"]:
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * warmup_cosine(epoch, epochs, *cfg["sched"])
        x_epoch = crop_flip(tx, gen) if cfg["aug"] else tx
        perm = torch.randperm(tx.shape[0], device=dev, generator=gen)[:n_train]
        net.train()
        total = torch.zeros((), device=dev)
        for i in range(0, n_train, b):
            idx = perm[i:i + b]
            xb = x_epoch[idx]
            total += step(xb.view(b, -1) if cfg["flat"] else xb, ty[idx])
        sync()
        t1 = time.perf_counter()
        net.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, n_test, b):
                xb = vx[i:i + b]
                logits = model(xb.view(b, -1) if cfg["flat"] else xb)
                correct += (logits.argmax(1) == vy[i:i + b]).sum()
        top1 = correct.item() / n_test
        t2 = time.perf_counter()
        loss = total.item() / (n_train // b)
        out["epoch_train_s"].append(t1 - t0)
        out["epoch_eval_s"].append(t2 - t1)
        out["epoch_top1"].append(top1)
        out["epoch_loss"].append(loss)
        print(f"epoch {epoch} | train_loss={loss:.4f} | test_top1={100 * top1:.2f}% "
              f"| train={t1 - t0:.3f}s | eval={t2 - t1:.3f}s")
    steady = out["epoch_train_s"][1:] or out["epoch_train_s"]
    out["ms_per_step"] = 1000.0 * (sum(steady) / len(steady)) / (n_train // b)
    out["best_top1"] = max(out["epoch_top1"])
    print(f"best test accuracy: {100 * out['best_top1']:.2f}% | "
          f"steady train step {out['ms_per_step']:.3f} ms (epochs >= 1)")
    print("RESULT " + json.dumps(out))


def run_gpt(args, dev):
    torch.manual_seed(args.seed)
    text = open(os.path.join(CACHE, "tinyshakespeare", "input.txt")).read()
    chars = sorted(set(text))
    ids = torch.tensor([chars.index(c) for c in text], dtype=torch.long)
    n_train = int(len(ids) * 0.9)
    train, val = ids[:n_train].to(dev), ids[n_train:].to(dev)
    seq, b, iters = 256, 64, args.iters or 5000
    warm, min_scale, base_lr = 100, 0.1, 1e-3
    net = GPT(vocab=len(chars)).to(dev)
    model = wrap(net, args.mode)
    decay = [p for n, p in net.named_parameters() if p.dim() >= 2 and n != "pos"]
    no_decay = [p for n, p in net.named_parameters() if p.dim() < 2 or n == "pos"]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=base_lr, betas=(0.9, 0.99), fused=dev.type == "cuda")
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    ar = torch.arange(seq, device=dev)

    def amp():  # `--amp bf16` = standard mixed precision: fp32 master weights
        return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=args.amp == "bf16")

    def batch(src, n):
        s = torch.randint(0, src.numel() - seq, (n,), device=dev, generator=gen)
        w = s[:, None] + ar[None, :]
        return src[w], src[w + 1]

    def lr_at(it):  # `AutoregressiveTrainer._lr_scale`
        if it < warm:
            return (it + 1) / warm
        prog = min(1.0, (it - warm) / max(1, iters - warm))
        return min_scale + (1 - min_scale) * 0.5 * (1 + math.cos(math.pi * prog))

    def step(it):
        for g in opt.param_groups:
            g["lr"] = base_lr * lr_at(it)
        x, y = batch(train, b)
        with amp():
            loss = F.cross_entropy(model(x).view(-1, len(chars)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        return loss.detach()

    vx, vy = batch(val, 256)

    def val_loss():
        net.eval()
        with torch.no_grad(), amp():
            tot = sum(F.cross_entropy(model(vx[i:i + b]).view(-1, len(chars)),
                                      vy[i:i + b].reshape(-1)) for i in range(0, 256, b))
        net.train()
        return (tot / (256 // b)).item()

    print(f"[gpt] mode={args.mode} tf32={args.tf32} amp={args.amp} iters={iters} torch={torch.__version__}")
    out = dict(model="gpt", mode=args.mode, tf32=args.tf32, amp=args.amp, batch=b, iters=iters)
    if args.bench_steps:
        net.train()
        for it in range(args.bench_steps):        # warmup: compile, capture
            step(it)
        sync()
        t0 = time.perf_counter()
        for it in range(args.bench_steps):
            step(it)
        sync()
        out["ms_per_step"] = 1000.0 * (time.perf_counter() - t0) / args.bench_steps
        print(f"steady train step {out['ms_per_step']:.3f} ms over {args.bench_steps} steps")
    else:
        net.train()
        sync()
        t0 = time.perf_counter()
        for it in range(iters):
            tl = step(it)
            if (it + 1) % 250 == 0 or it + 1 == iters:
                v = val_loss()
                print(f"  iter {it + 1}/{iters}  train={tl.item():.4f}  val={v:.4f}")
        sync()
        out["train_s"] = time.perf_counter() - t0
        out["final_val"] = v
        print(f"training time: {out['train_s']:.3f} s | final val {v:.4f}")
    print("RESULT " + json.dumps(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", choices=[*CLASSIFIERS, "gpt"])
    ap.add_argument("--mode", choices=["eager", "compile", "cudagraph"], default="eager")
    ap.add_argument("--tf32", choices=["on", "off"], default="on")
    ap.add_argument("--epochs", type=int, default=0, help="override (classifiers)")
    ap.add_argument("--iters", type=int, default=0, help="override (gpt)")
    ap.add_argument("--bench-steps", type=int, default=0,
                    help="gpt: time this many steps after as many warmup steps, no eval")
    ap.add_argument("--amp", choices=["off", "bf16"], default="off",
                    help="gpt: autocast(bfloat16) around forward + loss (fp32 master weights)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = args.tf32 == "on"
    torch.backends.cudnn.allow_tf32 = args.tf32 == "on"
    if args.model == "gpt":
        run_gpt(args, dev)
    else:
        run_classifier(args.model, args, dev)


if __name__ == "__main__":
    main()
