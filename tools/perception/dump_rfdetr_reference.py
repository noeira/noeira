"""RF-DETR-Seg-Small (Roboflow `rfdetr`, Apache-2.0) run on the CPU, every stage
dumped — the reference `noeira.perception.rfdetr` is gated against.

    pixi run -e detect ~/.cache/noeira/venvs/detcand/bin/python tools/perception/dump_rfdetr_reference.py \\
        [--image ~/.cache/noeira/sam3/ref/image.npy] [--out ~/.cache/noeira/rfdetr/ref] [--threshold 0.3]

(`rfdetr` lives in the detcand venv, not in pixi: `detector_eval.py`'s docstring.)
The picture is SAM 3's reference frame (NYU labelled 921, 640 x 480, uint8 HWC
RGB). ONE `predict(threshold)` call on `device="cpu"` (MPS samples the deformable
attention with different rounding), forward hooks catching:

    pixel_values.npy   [1, 3, 384, 384]  the network's input (stretched, normalised)
    bb_embed.npy       [4, 257, 384]     windowed embeddings (4 windows x (CLS + 256))
    bb_layer{i}.npy    [4, 257, 384]     each DINOv2 block's output, i = 0..11
    bb_taps.npy        [4, 384, 32, 32]  the four feature maps (blocks 2, 5, 8, 11; LN, CLS dropped)
    src.npy            [1, 256, 32, 32]  the projector's output (C2f + LN)
    enc_cls.npy        [1, 1024, 91]     two-stage class logits (group 0) on every token
                                         (`_call1`: the same head on the selected 100)
    enc_delta.npy      [1, 100, 4]       two-stage box deltas for the top 100
    dec_ref.npy        [1, 100, 4]       the decoder's reference boxes (cxcywh, normalised)
    dec_tgt.npy        [1, 100, 256]     its content queries
    query_pos.npy      [1, 100, 256]     ref_point_head(sine(ref))
    dec_layer{l}.npy   [1, 100, 256]     each decoder layer's raw output, l = 0..3
    hs.npy             [4, 1, 100, 256]  the normalised decoder outputs
    pred_logits.npy    [1, 100, 91]   pred_boxes.npy [1, 100, 4]   pred_masks.npy [1, 100, 96, 96]
    post_scores.npy, post_labels.npy (COCO category ids), post_boxes.npy [K, 4] xyxy pixels,
    post_masks.npy [K, 480, 640] uint8 — `predict`'s detections above `--threshold`
    meta.json
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image


def first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        for v in x:
            t = first_tensor(v)
            if t is not None:
                return t
    if hasattr(x, "tensors"):
        return x.tensors
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.expanduser("~/.cache/noeira/sam3/ref/image.npy"))
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/rfdetr/ref"))
    ap.add_argument("--threshold", type=float, default=0.3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from rfdetr import RFDETRSegSmall

    torch.set_grad_enabled(False)
    m = RFDETRSegSmall(device="cpu")
    net = m.model.model
    net.eval()
    caught = {}

    def save(name, t):
        """A module called more than once keeps every call: name, name_call1, ..."""
        v = t.detach().float().cpu().numpy().copy()
        k, c = name, 0
        while k in caught:
            c += 1
            k = f"{name}_call{c}"
        caught[k] = v

    hooks = []

    def out_hook(name):
        def h(mod, inp, out):
            t = first_tensor(out)
            if t is not None:
                save(name, t)
        return h

    bb = net.backbone[0]
    dino = bb.encoder.encoder
    hooks.append(net.backbone.register_forward_pre_hook(lambda mod, inp: save("pixel_values", first_tensor(inp[0]))))
    hooks.append(dino.embeddings.register_forward_hook(out_hook("bb_embed")))
    for i, layer in enumerate(dino.encoder.layer):
        hooks.append(layer.register_forward_hook(out_hook(f"bb_layer{i}")))

    def taps_hook(mod, inp, out):
        save("bb_taps", torch.cat([t for t in out], 0))
    hooks.append(bb.encoder.register_forward_hook(taps_hook))
    hooks.append(bb.projector.register_forward_hook(out_hook("src")))
    tr = net.transformer
    hooks.append(tr.enc_out_class_embed[0].register_forward_hook(out_hook("enc_cls")))
    hooks.append(tr.enc_out_bbox_embed[0].register_forward_hook(out_hook("enc_delta")))

    def dec_pre(mod, args, kwargs):
        save("dec_tgt", args[0] if args else kwargs["tgt"])
        ref = kwargs.get("refpoints_unsigmoid")
        if ref is None and len(args) > 4:
            ref = args[4]
        if ref is not None:
            save("dec_ref", ref)
    hooks.append(tr.decoder.register_forward_pre_hook(dec_pre, with_kwargs=True))
    hooks.append(tr.decoder.ref_point_head.register_forward_hook(out_hook("query_pos")))
    for l, layer in enumerate(tr.decoder.layers):
        hooks.append(layer.register_forward_hook(out_hook(f"dec_layer{l}")))
    hooks.append(tr.decoder.register_forward_hook(out_hook("hs")))

    def net_hook(mod, inp, out):
        for k in ("pred_logits", "pred_boxes", "pred_masks"):
            save(k, out[k])
    hooks.append(net.register_forward_hook(net_hook))

    img = np.load(a.image).astype(np.uint8)
    det = m.predict(Image.fromarray(img), threshold=a.threshold)
    for h in hooks:
        h.remove()
    for k, v in caught.items():
        np.save(os.path.join(a.out, k + ".npy"), v)
    np.save(os.path.join(a.out, "post_scores.npy"), np.asarray(det.confidence, dtype=np.float32))
    np.save(os.path.join(a.out, "post_labels.npy"), np.asarray(det.class_id, dtype=np.int64))
    np.save(os.path.join(a.out, "post_boxes.npy"), np.asarray(det.xyxy, dtype=np.float32))
    masks = det.mask if det.mask is not None else np.zeros((0,) + img.shape[:2], bool)
    np.save(os.path.join(a.out, "post_masks.npy"), masks.astype(np.uint8))
    meta = {"image": a.image, "image_shape": list(img.shape), "threshold": a.threshold,
            "dumped": sorted(caught), "detections": int(len(det.confidence)),
            "labels": [int(c) for c in det.class_id], "scores": [float(s) for s in det.confidence]}
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    for k in sorted(caught):
        print(f"{k:14s} {caught[k].shape}")
    print("detections:", list(zip(meta["labels"], [round(s, 3) for s in meta["scores"]])))


if __name__ == "__main__":
    main()
