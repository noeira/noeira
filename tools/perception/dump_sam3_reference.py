#!/usr/bin/env python3
"""SAM 3's image path, run by transformers (`Sam3Model`, fp32, CPU), every stage
dumped — the reference the Mojo port (`noeira/perception/sam3/`) is gated
against (docs/NN_MODEL_PORTS_PLAN.md, SAM 3).

    pixi run -e detect python tools/perception/dump_sam3_reference.py \
        --image ~/.cache/noeira/d1/nyu/nyu_0921.png [--out ~/.cache/noeira/sam3/ref] [--words "door,table,sofa,bed"]

Needs `facebook/sam3` in the Hugging Face cache (gated, Meta's licence) and
transformers 5 (the `detect` env). Float32 on the CPU, deterministic; ~3.4 GB
of weights.

Written to `<out>/`:

    image.npy                 the picture as read, [H, W, 3] uint8
    pixel_values.npy          the processor's output [3, 1008, 1008] (resize to 1008x1008, (x/255 - .5)/.5)
    vit_embed.npy             patch embedding + tiled position embedding  [5184, 1024]
    vit_ln_pre.npy            after the backbone's `layer_norm` (applied BEFORE the layers)  [5184, 1024]
    vit_layer<l>.npy          layer l's output, l in LAYERS  [5184, 1024]  (31 = the backbone's output)
    fpn<i>.npy                the neck's level i (scale 4, 2, 1, 0.5)  [256, h, w]
    fpn_pos<i>.npy            its sine position encoding  [256, h, w]
    meta.json                 shapes, the layers dumped, the words
    words/<w>/ids.npy, mask.npy          the processor's 32 ids and attention mask
    words/<w>/text_layer<l>.npy          CLIP text layer l's output [32, 1024], l in TEXT_LAYERS
    words/<w>/text_final.npy             after `final_layer_norm` [32, 1024]
    words/<w>/text_features.npy          SAM 3's `text_projection` of it [32, 256] (`get_text_features`)
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

LAYERS = (0, 1, 6, 7, 8, 15, 23, 30, 31)
"""Windowed (0, 1, 6), the first global (7) and the windowed one after it
(8), every global (7, 15, 23, 31), the last windowed (30): enough to localise a
divergence to a layer kind."""


TEXT_LAYERS = (0, 1, 11, 23)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/sam3/ref"))
    ap.add_argument("--words", default="")
    ap.add_argument("--text-only", action="store_true", help="skip the vision path (already dumped)")
    args = ap.parse_args()

    from transformers import Sam3Model, Sam3Processor

    torch.set_grad_enabled(False)
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float32).eval()
    os.makedirs(args.out, exist_ok=True)

    if args.words:
        dump_words(model, proc, args)
    if args.text_only:
        return
    img = Image.open(args.image).convert("RGB")
    np.save(os.path.join(args.out, "image.npy"), np.asarray(img, dtype=np.uint8))
    pv = proc(images=img, return_tensors="pt")["pixel_values"]
    np.save(os.path.join(args.out, "pixel_values.npy"), pv[0].float().numpy())

    vit = model.vision_encoder.backbone
    cap = {}
    hooks = [
        vit.embeddings.register_forward_hook(lambda m, i, o: cap.__setitem__("embed", o)),
        vit.layer_norm.register_forward_hook(lambda m, i, o: cap.__setitem__("ln_pre", o)),
    ]
    for li in LAYERS:
        hooks.append(vit.layers[li].register_forward_hook(
            lambda m, i, o, li=li: cap.__setitem__(li, o[0] if isinstance(o, tuple) else o)))
    ve = model.get_vision_features(pixel_values=pv)
    for h in hooks:
        h.remove()

    def save(name, t):
        np.save(os.path.join(args.out, name + ".npy"), t.detach().float().contiguous().numpy())

    save("vit_embed", cap["embed"][0])
    save("vit_ln_pre", cap["ln_pre"][0].reshape(-1, cap["ln_pre"].shape[-1]))
    for li in LAYERS:
        save(f"vit_layer{li}", cap[li][0].reshape(-1, cap[li].shape[-1]))
    shapes = {}
    for i, (f, p) in enumerate(zip(ve.fpn_hidden_states, ve.fpn_position_encoding)):
        save(f"fpn{i}", f[0])
        save(f"fpn_pos{i}", p[0])
        shapes[f"fpn{i}"] = list(f[0].shape)
    json.dump({"image": args.image, "image_shape": list(np.asarray(img).shape), "pixel_values": list(pv[0].shape),
               "layers": list(LAYERS), "fpn": shapes, "words": [w for w in args.words.split(",") if w]},
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print("vision:", {k: tuple(v.shape) for k, v in cap.items() if not isinstance(k, int)}, "fpn", shapes)


def dump_words(model, proc, args):
    tm = model.text_encoder.text_model
    for w in [w for w in args.words.split(",") if w]:
        out = os.path.join(args.out, "words", w.replace(" ", "_"))
        os.makedirs(out, exist_ok=True)
        ti = proc(text=w, return_tensors="pt")
        cap = {}
        hooks = [tm.encoder.layers[l].register_forward_hook(
            lambda m, i, o, l=l: cap.__setitem__(l, o[0] if isinstance(o, tuple) else o)) for l in TEXT_LAYERS]
        tf = model.get_text_features(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"])
        for h in hooks:
            h.remove()
        np.save(os.path.join(out, "ids.npy"), ti["input_ids"][0].numpy().astype(np.int64))
        np.save(os.path.join(out, "mask.npy"), ti["attention_mask"][0].numpy().astype(np.int64))
        for l in TEXT_LAYERS:
            np.save(os.path.join(out, f"text_layer{l}.npy"), cap[l][0].float().numpy())
        np.save(os.path.join(out, "text_final.npy"), tf.last_hidden_state[0].float().numpy())
        np.save(os.path.join(out, "text_features.npy"), tf.pooler_output[0].float().numpy())
        print("  word", repr(w), int(ti["attention_mask"].sum()), "tokens")


if __name__ == "__main__":
    main()
