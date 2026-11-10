#!/usr/bin/env python3
"""Gradients through d1-2B's own code (torch autograd, fp32) with LoRA adapters,
the reference the Mojo backward of LFM3 is gated against (docs/NN_MODEL_PORTS_PLAN.md §12.5, step 5).

    python tools/d1/dump_d1_grad_reference.py [--model ~/.cache/noeira/local-ai/d1/d1-2B] \
        [--ref ~/.cache/noeira/d1/ref] [--out ~/.cache/noeira/d1/grad]

Same environment as `dump_d1_reference.py` (transformers 5.19 + torchvision; on
the Orin `~/Documents/noeira-parakeet/d1ref-venv`). It reads the rows that
script dumped (ids, each token's question, state length, slots) and does NOT
build d1's engine (whose `fuse_projections` would merge q/k/v/gate and w1/w3).

LoRA as Liquid's cookbook trains LFM3: rank 16, alpha 32 (scale 2), on every
`q_proj k_proj v_proj o_proj in_proj out_proj w1 w2 w3` of the language model;
`y = x W^T + 2 (x A^T) B^T`, A [16, in] and B [out, 16] drawn N(0, 0.02^2) from
seed 0 in module order — B NONZERO, so A's gradient is not trivially zero.
Loss: `-sum_s log_softmax(lm_head(h[slot_s]))[target_s]`, target_s the THIRD most
likely token at slot s under the adapted model (a well-conditioned loss; see the
comment at `targets`). Per case (`<out>/<case>/`):

    lora.safetensors     every adapter's A and B: `layers.<i>.<module>.lora_A|lora_B`
    grads.safetensors    d loss / d A, d B (same names) and d loss / d embeddings (`embeds`, [T, 2048])
    slot_logz.npy        the adapted model's log-softmax at the slots [S, 128000]
    layer<l>.npy         the adapted model's layer outputs [T, 2048]
    meta.json            loss, targets, rank, scale, the row it read
"""

import argparse
import importlib
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModel

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "w1", "w2", "w3")
RANK = 16
ALPHA = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/.cache/noeira/local-ai/d1/d1-2B"))
    ap.add_argument("--ref", default=os.path.expanduser("~/.cache/noeira/d1/ref"))
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/d1/grad"))
    ap.add_argument("--cases", default="room_en,single_fr")
    args = ap.parse_args()

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lm = model.model.language_model
    slots_mod = importlib.import_module(type(model).__module__.rsplit(".", 1)[0] + ".slots")

    # deterministic adapters, module order
    gen = torch.Generator().manual_seed(0)
    scale = ALPHA / RANK
    lora = {}
    for li, layer in enumerate(lm.layers):
        for mname, mod in layer.named_modules():
            leaf = mname.rsplit(".", 1)[-1]
            if leaf not in TARGETS or not isinstance(mod, torch.nn.Linear):
                continue
            key = f"layers.{li}.{mname}"
            a = (torch.randn(RANK, mod.in_features, generator=gen) * 0.02).requires_grad_(True)
            b = (torch.randn(mod.out_features, RANK, generator=gen) * 0.02).requires_grad_(True)
            lora[key] = (a, b)

            def fwd(x, mod=mod, a=a, b=b):
                return torch.nn.functional.linear(x, mod.weight) + scale * torch.nn.functional.linear(
                    torch.nn.functional.linear(x, a), b)
            mod.forward = fwd
    print(f"{len(lora)} adapters, rank {RANK}, scale {scale}")

    for case in args.cases.split(","):
        src = os.path.join(args.ref, case)
        ids = np.load(os.path.join(src, "row0_ids.npy")).tolist()
        question = np.load(os.path.join(src, "row0_question.npy")).tolist()
        meta = json.load(open(os.path.join(src, "row0_meta.json")))
        slots = [p for own in meta["slots"] for p in own]
        cap = {}

        def grab(m, i, o):
            o.requires_grad_(True)
            o.retain_grad()
            cap["embeds"] = o
            return o
        h = lm.embed_tokens.register_forward_hook(grab)
        layer_out = {}
        lh = [layer.register_forward_hook(
            lambda m, i, o, li=li: layer_out.__setitem__(li, (o[0] if isinstance(o, tuple) else o).detach()))
            for li, layer in enumerate(lm.layers)]
        for a, b in lora.values():
            a.grad = None
            b.grad = None
        t = torch.tensor([ids])
        rows = slots_mod.slot_logits(model, t, torch.ones_like(t), torch.tensor(slots), torch.tensor([meta["state_len"]]),
                                     torch.tensor([question]), torch.zeros(len(slots), dtype=torch.long))
        h.remove()
        for x in lh:
            x.remove()
        logz = torch.log_softmax(rows.codes.float(), dim=-1)
        # The THIRD most likely token: (1 - p) ~ 1, so the gradient is O(1) and a
        # log-probability difference at fp32's level does not rescale it. With the
        # argmax (p ~ 0.98) every gradient is proportional to 1 - p ~ 0.02, and a
        # 3e-4 difference in that log-probability moved them all by ~2 %.
        targets = logz.topk(3, dim=-1).indices[:, 2].tolist()
        loss = -logz[torch.arange(len(slots)), torch.tensor(targets)].sum()
        loss.backward()

        out = os.path.join(args.out, case)
        os.makedirs(out, exist_ok=True)
        save_file({f"{k}.lora_A": a.detach().contiguous() for k, (a, _) in lora.items()}
                  | {f"{k}.lora_B": b.detach().contiguous() for k, (_, b) in lora.items()},
                  os.path.join(out, "lora.safetensors"))
        grads = {f"{k}.lora_A": a.grad.contiguous() for k, (a, _) in lora.items()}
        grads |= {f"{k}.lora_B": b.grad.contiguous() for k, (_, b) in lora.items()}
        grads["embeds"] = cap["embeds"].grad[0].contiguous()
        save_file(grads, os.path.join(out, "grads.safetensors"))
        np.save(os.path.join(out, "slot_logz.npy"), logz.detach().numpy())
        for li, o in layer_out.items():  # the adapted forward, layer by layer (to localise a divergence)
            np.save(os.path.join(out, f"layer{li}.npy"), o[0].numpy())
        json.dump({"case": case, "loss": loss.item(), "targets": targets, "rank": RANK, "scale": scale,
                   "length": len(ids), "state_len": meta["state_len"], "slots": slots},
                  open(os.path.join(out, "meta.json"), "w"), indent=1)
        gmax = max(g.abs().max().item() for g in grads.values())
        print(f"  {case}: {len(ids)} tokens, loss {loss.item():.6f}, targets {targets}, |grad| max {gmax:.3e}")


if __name__ == "__main__":
    main()
