#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | PEFT, on both ends of noeira's LoRA
# +--------------------------------------------------------------------------+ #
"""Gate `noeira/nn/peft/` against PEFT itself, in both directions.

    pixi run -e act-ref python tools/peft/lora_peft_reference.py \
        --out tests/fixtures/peft                # 1. PEFT writes (committed)
    pixi run mojo run -I . tests/nn/peft/test_peft_io.mojo
                                                 # 2. noeira reads it, matches
                                                 #    PEFT's numbers, and writes
                                                 #    its own to /tmp/peft_ours
    pixi run -e act-ref python tools/peft/lora_peft_reference.py \
        --verify /tmp/peft_ours                  # 3. PEFT loads noeira's

## The model

`Net` is one `nn.Linear(13, 7)` at the module path `proj`, so PEFT's keys are
`base_model.model.proj.lora_A.weight` ([R, IN]) and `...lora_B.weight`
([OUT, R]). Two adapters, both r = 4 and lora_alpha = 10: plain (scale 2.5)
and rsLoRA (scale 10 / √4 = 5), so a wrong scale formula cannot pass both.

⚠ PEFT initialises lora_B to ZERO. Gating dA or dX against a zero B compares
zeros (dA = xᵀ (s g B) = 0), so lora_B is overwritten with seeded values
before anything is dumped.

## Files (`--out`)

    base.safetensors          proj.weight [7, 13], proj.bias [7] (torch layout)
    adapter/                  PEFT's save_pretrained (fp32)
    adapter_rslora/           the same, use_rslora = True
    adapter_bf16/             `adapter` saved from a bf16 copy: noeira's bf16
                              writer must round exactly as torch does
    io.safetensors            x [5, 13], g [5, 7], and from autograd, per
                              adapter: y, dx, dA ([4, 13]), dB ([7, 4])

`--verify DIR` reads what noeira's gate wrote: `DIR/base.safetensors`,
`DIR/adapter/` and `DIR/io.safetensors` (x and noeira's y). It checks that the
adapter's keys are exactly the ones PEFT itself would save for this model, that
the config round-trips, and that PEFT's forward over noeira's adapter equals
noeira's own forward.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict

IN, OUT, R, ALPHA, T = 13, 7, 4, 10, 5


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(IN, OUT)

    def forward(self, x):
        return self.proj(x)


def base_net(seed=0):
    torch.manual_seed(seed)
    return Net().float()


def wrap(net, rslora):
    cfg = LoraConfig(
        r=R,
        lora_alpha=ALPHA,
        target_modules=["proj"],
        use_rslora=rslora,
        lora_dropout=0.0,
        bias="none",
    )
    return get_peft_model(net, cfg)


def set_b(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        b = model.base_model.model.proj.lora_B["default"].weight
        b.copy_(torch.randn(b.shape, generator=g) * 0.5)


def run(model, x, g):
    x = x.clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    y = model(x)
    (y * g).sum().backward()
    lin = model.base_model.model.proj
    return {
        "y": y.detach(),
        "dx": x.grad.detach(),
        "dA": lin.lora_A["default"].weight.grad.detach(),
        "dB": lin.lora_B["default"].weight.grad.detach(),
    }


def out(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    net = base_net()
    save_file(
        {
            "proj.weight": net.proj.weight.detach().contiguous(),
            "proj.bias": net.proj.bias.detach().contiguous(),
        },
        str(path / "base.safetensors"),
    )
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(T, IN, generator=gen)
    g = torch.randn(T, OUT, generator=gen)
    io = {"x": x, "g": g}
    for name, rslora, seed in (("adapter", False, 2), ("adapter_rslora", True, 3)):
        model = wrap(base_net(), rslora)
        set_b(model, seed)
        model.save_pretrained(str(path / name))
        (path / name / "README.md").unlink()  # PEFT's model card, not data
        res = run(model, x, g)
        suffix = "" if name == "adapter" else "_rs"
        for k, v in res.items():
            io[k + suffix] = v.float().contiguous()
        if name == "adapter":
            model.to(torch.bfloat16).save_pretrained(str(path / "adapter_bf16"))
            (path / "adapter_bf16" / "README.md").unlink()
    save_file(io, str(path / "io.safetensors"))
    print("wrote", path)


def verify(path):
    path = Path(path)
    net = Net()
    base = load_file(str(path / "base.safetensors"))
    with torch.no_grad():
        net.proj.weight.copy_(base["proj.weight"])
        net.proj.bias.copy_(base["proj.bias"])
    adir = path / "adapter"
    model = PeftModel.from_pretrained(net, str(adir))
    cfg = json.loads((adir / "adapter_config.json").read_text())
    fails = 0

    def check(ok, what):
        nonlocal fails
        print(("  PASS  " if ok else "  FAIL  ") + what)
        if not ok:
            fails += 1

    ours = load_file(str(adir / "adapter_model.safetensors"))
    want = get_peft_model_state_dict(model)
    check(
        sorted(ours) == sorted(want),
        f"keys are exactly PEFT's: ours {sorted(ours)} vs {sorted(want)}",
    )
    for k in want:
        if k in ours:
            check(
                tuple(ours[k].shape) == tuple(want[k].shape)
                and torch.equal(ours[k].float(), want[k].float()),
                f"{k} {tuple(ours[k].shape)} {ours[k].dtype} loaded bit-exactly",
            )
    lc = model.peft_config["default"]
    check(
        lc.r == cfg["r"] and lc.lora_alpha == cfg["lora_alpha"]
        and lc.use_rslora == cfg.get("use_rslora", False)
        and sorted(lc.target_modules) == sorted(cfg["target_modules"]),
        f"config: r {lc.r} alpha {lc.lora_alpha} rslora {lc.use_rslora}"
        f" targets {sorted(lc.target_modules)}",
    )
    io = load_file(str(path / "io.safetensors"))
    with torch.no_grad():
        y = model(io["x"])
    yn = io["y"]
    rel = ((y - yn).abs().max() / yn.abs().max()).item()
    lora = model.base_model.model.proj
    s = lora.scaling["default"]
    check(rel < 1e-6, f"PEFT's forward over noeira's adapter vs noeira's: max rel {rel:.3e}")
    nb = lora.lora_B["default"].weight.abs().max().item()
    check(nb > 0.05, f"the adapter is not a no-op (max |B| {nb:.3f}, scale {s})")
    print("FAILED" if fails else "PASSED")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--verify")
    a = ap.parse_args()
    if a.out:
        out(a.out)
    if a.verify:
        sys.exit(verify(a.verify))


if __name__ == "__main__":
    main()
