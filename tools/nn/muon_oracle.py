#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | Muon oracle: torch.optim.Muon, and its algorithm in float64
# +--------------------------------------------------------------------------+ #
"""Reference tensors for `tests/nn/test_muon.mojo`.

    uv run --no-project --python 3.12 --with torch --with numpy \\
        python tools/nn/muon_oracle.py --out /tmp/muon_oracle

Two matrices, three steps each, `adjust_lr_fn="match_rms_adamw"`:

    a   [IN 12, OUT 20]   weight decay 0      (the Gram matrix on the IN side)
    b   [IN 24, OUT 10]   weight decay 0.1    (on the OUT side)

For every step two results are written:

    <tag>_w<k>_torch.npy   `torch.optim.Muon` itself: its Newton-Schulz
                           products run in bfloat16
    <tag>_w<k>_f64.npy     torch's OWN update code with every tensor in
                           float64: `_zeropower_via_newtonschulz`'s source
                           with `torch.bfloat16` replaced by `torch.float64`
                           (asserted), and `_single_tensor_muon` unchanged

The second is the gate's reference: noeira's Newton-Schulz is fp32, so it
should sit within fp32 rounding of it and within bfloat16's noise of the
first. Weights and gradients are rounded to float32 first, and every file is
in noeira's layout: `[IN, OUT]` row-major, torch's weight transposed.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import torch
from torch.optim import _muon

LR = 1e-2


def f64_newton_schulz():
    """torch's `_zeropower_via_newtonschulz`, recompiled to work in float64."""
    src = inspect.getsource(_muon._zeropower_via_newtonschulz)
    assert src.count("torch.bfloat16") == 1, "torch's Newton-Schulz no longer casts to bfloat16 in one place"
    scope = dict(vars(_muon))
    exec(src.replace("torch.bfloat16", "torch.float64"), scope)
    return scope["_zeropower_via_newtonschulz"]


def save(path: Path, w: torch.Tensor) -> None:
    """torch's [OUT, IN] weight as noeira's [IN, OUT], C order (a bare `.T` view would be saved column-major)."""
    np.save(path, np.ascontiguousarray(w.double().numpy().T))


def run(tag: str, n_in: int, n_out: int, wd: float, out: Path, gen: torch.Generator) -> None:
    w0 = (0.3 * torch.randn(n_out, n_in, generator=gen)).float()
    grads = [(torch.randn(n_out, n_in, generator=gen) * s).float() for s in (1.0, 0.3, 3.0)]
    save(out / f"{tag}_w0.npy", w0)

    # torch.optim.Muon as shipped
    p = torch.nn.Parameter(w0.clone())
    opt = torch.optim.Muon([p], lr=LR, weight_decay=wd, adjust_lr_fn="match_rms_adamw")
    # torch's update code in float64
    p64 = w0.double().clone()
    buf64 = torch.zeros_like(p64)
    saved = _muon._zeropower_via_newtonschulz
    ns64 = f64_newton_schulz()
    d = opt.defaults
    for k, g in enumerate(grads, start=1):
        save(out / f"{tag}_g{k}.npy", g)
        p.grad = g.clone()
        opt.step()
        save(out / f"{tag}_w{k}_torch.npy", p.detach())
        _muon._zeropower_via_newtonschulz = ns64
        try:
            _muon._single_tensor_muon(
                [p64], [g.double()], [buf64], lr=LR, weight_decay=wd, momentum=d["momentum"], nesterov=d["nesterov"],
                ns_coefficients=d["ns_coefficients"], ns_steps=d["ns_steps"], eps=d["eps"],
                adjust_lr_fn="match_rms_adamw", has_complex=False,
            )
        finally:
            _muon._zeropower_via_newtonschulz = saved
        save(out / f"{tag}_w{k}_f64.npy", p64)
        step = (p64 - (w0.double() if k == 1 else prev)).abs().max().item()
        gap = (p.detach().double() - p64).abs().max().item()
        print(f"{tag} step {k}: largest move {step:.3e} | torch (bf16) vs float64 {gap:.3e} = {gap / step:.2%} of it")
        prev = p64.clone()
    print(f"{tag}: defaults momentum {d['momentum']} nesterov {d['nesterov']} ns_steps {d['ns_steps']} coefficients {d['ns_coefficients']} eps {d['eps']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(7)
    print("torch", torch.__version__, "| lr", LR)
    run("a", 12, 20, 0.0, out, gen)
    run("b", 24, 10, 0.1, out, gen)


if __name__ == "__main__":
    main()
