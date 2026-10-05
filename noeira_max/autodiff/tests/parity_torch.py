"""Torch side of the model-level parity tests (``act-ref`` env, never imports
MAX): the same models, from the same initial weights, trained on the same
batches, in float64, with SGD (M1) or with the twin's AdamW recipe (M2:
parameter groups, warmup + cosine schedule, global-norm clipping).

    env -u LD_PRELOAD .pixi/envs/act-ref/bin/python \\
        noeira_max/autodiff/tests/parity_torch.py {mlp|gpt} IN.npz OUT.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# MAX stores layer_norm's epsilon as a float32 constant; use the same value.
EPS = float(np.float32(1e-5))


def mlp_loss(p, x, y):
    h = x
    for i in range(3):
        h = h @ p[f"w{i}"] + p[f"b{i}"]
        if i < 2:
            h = torch.relu(h)
    return F.cross_entropy(h, y)


def gpt_loss(p, idx, targets, cfg):
    vocab, seq, d, heads, layers = cfg
    b, t = idx.shape
    x = p["wte"][idx] + p["wpe"]
    for l in range(layers):
        h = F.layer_norm(x, (d,), p[f"h{l}.ln1.w"], p[f"h{l}.ln1.b"], EPS)
        q, k, v = (h @ p[f"h{l}.qkv.w"] + p[f"h{l}.qkv.b"]).split(d, dim=2)
        q, k, v = (z.view(b, t, heads, d // heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + y.transpose(1, 2).reshape(b, t, d) @ p[f"h{l}.proj.w"] + p[f"h{l}.proj.b"]
        h = F.layer_norm(x, (d,), p[f"h{l}.ln2.w"], p[f"h{l}.ln2.b"], EPS)
        h = F.gelu(h @ p[f"h{l}.fc1.w"] + p[f"h{l}.fc1.b"], approximate="tanh")
        x = x + h @ p[f"h{l}.fc2.w"] + p[f"h{l}.fc2.b"]
    x = F.layer_norm(x, (d,), p["lnf.w"], p["lnf.b"], EPS)
    logits = x @ p["wte"].T  # tied head
    return F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))


def lr_scale(it, warmup, total, min_scale):
    """The twin's ``lr_at``; ``warmup == 0`` means a constant rate."""
    if warmup == 0:
        return 1.0
    if it < warmup:
        return (it + 1) / warmup
    prog = min(1.0, (it - warmup) / max(1, total - warmup))
    return min_scale + (1 - min_scale) * 0.5 * (1 + math.cos(math.pi * prog))


def train_sgd(loss_fn, params, data, names, steps, lr):
    step = torch.func.grad_and_value(loss_fn)
    losses = []
    for s in range(steps):
        grads, loss = step(params, torch.from_numpy(data[f"x{s}"]), torch.from_numpy(data[f"y{s}"]))
        losses.append(loss.item())
        params = {n: params[n] - lr * grads[n] for n in names}
    return losses, params


def train_adamw(loss_fn, params, data, names, steps, lr):
    """The twin's loop: set the scheduled rate, backward, clip, step."""
    params = {n: p.clone().requires_grad_(True) for n, p in params.items()}
    decay = [bool(d) for d in data["decay"]]
    groups = [
        {"params": [params[n] for n, d in zip(names, decay) if d],
         "weight_decay": float(data["weight_decay"])},
        {"params": [params[n] for n, d in zip(names, decay) if not d], "weight_decay": 0.0},
    ]
    b1, b2 = (float(b) for b in data["betas"])
    opt = torch.optim.AdamW(
        [g for g in groups if g["params"]], lr=lr, betas=(b1, b2),
        eps=float(data["eps"]), foreach=False,
    )
    warmup, total, min_scale = int(data["warmup"]), int(data["total"]), float(data["min_scale"])
    clip = float(data["clip"])
    losses = []
    for s in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * lr_scale(s, warmup, total, min_scale)
        loss = loss_fn(params, torch.from_numpy(data[f"x{s}"]), torch.from_numpy(data[f"y{s}"]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(list(params.values()), clip)
        opt.step()
        losses.append(loss.item())
    return losses, {n: p.detach() for n, p in params.items()}


def main(model: str, npz_path: str, out_path: str) -> None:
    data = np.load(npz_path)
    names = [str(n) for n in data["names"]]
    params = {n: torch.from_numpy(data[f"init.{n}"].copy()) for n in names}
    steps, lr = int(data["steps"]), float(data["lr"])
    if model == "mlp":
        loss_fn = mlp_loss
    else:
        cfg = tuple(int(v) for v in data["config"])
        loss_fn = lambda p, x, y: gpt_loss(p, x, y, cfg)  # noqa: E731

    optimizer = str(data["optim"]) if "optim" in data else "sgd"
    train = train_adamw if optimizer == "adamw" else train_sgd
    losses, params = train(loss_fn, params, data, names, steps, lr)

    ours = data["losses"]
    param_diff = {
        n: float(np.abs(data[f"final.{n}"] - params[n].numpy()).max()
                 / max(np.abs(params[n].numpy()).max(), 1e-300))
        for n in names
    }
    Path(out_path).write_text(json.dumps({
        "torch_losses": losses,
        "loss_rel_diff": [abs(a - b) / abs(b) for a, b in zip(ours, losses)],
        "param_rel_diff": param_diff,
    }, indent=1))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main(*sys.argv[1:4])
