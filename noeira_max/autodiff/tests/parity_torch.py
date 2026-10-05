"""Torch side of the model-level parity test (``act-ref`` env, never imports
MAX): the same models, from the same initial weights, trained with SGD on the
same batches, in float64.

    env -u LD_PRELOAD .pixi/envs/act-ref/bin/python \\
        noeira_max/autodiff/tests/parity_torch.py {mlp|gpt} IN.npz OUT.json
"""

from __future__ import annotations

import json
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

    step = torch.func.grad_and_value(loss_fn)
    losses = []
    for s in range(steps):
        x = torch.from_numpy(data[f"x{s}"])
        y = torch.from_numpy(data[f"y{s}"])
        grads, loss = step(params, x, y)
        losses.append(loss.item())
        params = {n: params[n] - lr * grads[n] for n in names}

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
