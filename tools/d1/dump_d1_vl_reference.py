#!/usr/bin/env python3
"""d1-2B's System One WITH PICTURES, run by its own code in float32, every stage of
the vision path dumped — the reference the Mojo LFM3-VL vision tower is gated
against (docs/NN_MODEL_PORTS_PLAN.md §12.5, step 6).

    python tools/d1/dump_d1_vl_reference.py --images DIR [--model ~/.cache/noeira/local-ai/d1/d1-2B] \
        [--out ~/.cache/noeira/d1/ref_vl]

Same environment as `dump_d1_reference.py` (transformers 5.19 + torchvision: on the
Orin `~/Documents/noeira-parakeet/d1ref-venv`). `DIR` holds NYU room frames
(`tools/perception/detector_latency.py --export-nyu`): nyu_0390.png, nyu_0740.png,
nyu_0921.png (640x480).

Per case (`<out>/<case>/`):

    answers.json            `system_one`'s output with the pictures in `media`
    prompt.json             the state prefix (WITH the pictures' markup) and each suffix
    image<k>.npy            picture k as d1 read it (after its pixel cap) [H, W, 3] uint8
    pixel_values.npy        the processor's patches, every picture [N, 1536] (merge-block order)
    grid_thw.npy            each picture's (t, h, w) patch grid [K, 3] int64
    vis_embed.npy           the patchifier + `embedding_norm`            [N, 1024]
    vis_layer<l>.npy        vision layer l's output, l = 0..13           [N, 1024]
    vis_proj.npy            the projector's output (the image tokens)    [N / 4, 2048]
    row<r>_*                as `dump_d1_reference.py`: ids, question, meta, embed (the
                            text embeddings WITH the image features in place), layer<l>,
                            final, slot_logz

Cases: `vl_one`, one 640x480 picture and three questions about it; `vl_two`, a
512x384 centre crop and a 640x480 picture (two grids, the tower's block-diagonal
attention, the features' scatter order) under a JSON state.
"""

import argparse
import importlib
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel


def cases(images_dir):
    def img(name, crop=None):
        im = Image.open(os.path.join(images_dir, name)).convert("RGB")
        if crop:
            w, h = im.size
            cw, ch = crop
            im = im.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))
        return im

    return {
        "vl_one": ("The robot's head camera sees this.", [img("nyu_0390.png")], {
            "door_open": {"type": "noul", "instructions": "Is there an open door in the picture?"},
            "furniture": {"type": "choice", "instructions": "Which of these is in the picture?",
                          "criteria": {"sofa": "a sofa", "bed": "a bed", "fridge": "a fridge", "car": "a car"}},
            "light": {"type": "score", "instructions": "How bright is the room?",
                      "criteria": ["dark", "dim", "bright"]},
        }),
        "vl_two": ({"robot": {"at": "door"}, "heard": "go to the bed", "pictures": "first: ahead, second: behind"},
                   [img("nyu_0921.png", (512, 384)), img("nyu_0740.png")], {
            "bed_first": {"type": "noul", "instructions": "Is there a bed in the first picture?"},
            "where": {"type": "choice", "instructions": "Where should the robot walk?",
                      "criteria": {"ahead": "walk ahead", "behind": "turn around and walk",
                                   "stay": "do not walk anywhere"}},
        }),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/.cache/noeira/local-ai/d1/d1-2B"))
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/d1/ref_vl"))
    ap.add_argument("--images", required=True)
    ap.add_argument("--cases", default="vl_one,vl_two")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32)
    model.eval()
    eng = model.engine
    lm = model.model.language_model
    tower = model.model.vision_tower
    proj = model.model.multi_modal_projector
    pkg = type(model).__module__.rsplit(".", 1)[0]  # the package transformers loaded d1's remote code into
    slot_logits = importlib.import_module(pkg + ".slots").slot_logits
    media_list = importlib.import_module(pkg + ".media").media_list
    all_cases = cases(args.images)
    print(f"d1 at {args.model}: pixel cap {eng.max_pixels}, tower {len(tower.layers)} layers")

    for name in args.cases.split(","):
        state, pictures, questions = all_cases[name]
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)
        answers = model.system_one(state, questions, media=pictures)
        with open(os.path.join(out, "answers.json"), "w") as f:
            json.dump(answers, f, ensure_ascii=False, indent=1)

        media = [eng.media._decoded(m) for m in media_list(pictures)]
        for k, im in enumerate(media):
            np.save(os.path.join(out, f"image{k}.npy"), np.asarray(im.convert("RGB"), dtype=np.uint8))
        qs = [eng.ask(q) for q in questions.values()]
        prefix, _ = eng._prefix(state, media_list(pictures))
        suffixes, offsets, _ = eng._suffixes(qs)
        with open(os.path.join(out, "prompt.json"), "w") as f:
            json.dump({"prefix": prefix, "suffixes": suffixes, "slot_offsets": offsets,
                       "names": list(questions)}, f, ensure_ascii=False, indent=1)

        seqs = eng._sequences([(state, qs, pictures)])
        for r, (_, k, ids, state_len, question, ends, inputs) in enumerate(seqs):
            cap = {}
            hooks = [tower.embedding_norm.register_forward_hook(lambda m, i, o: cap.__setitem__("vis_embed", o)),
                     proj.register_forward_hook(lambda m, i, o: cap.__setitem__("vis_proj", o)),
                     lm.embed_tokens.register_forward_hook(lambda m, i, o: cap.__setitem__("tok_embed", o)),
                     lm.embedding_norm.register_forward_hook(lambda m, i, o: cap.__setitem__("final", o))]
            for li, layer in enumerate(tower.layers):
                hooks.append(layer.register_forward_hook(
                    lambda m, i, o, li=li: cap.__setitem__(f"vis{li}", o[0] if isinstance(o, tuple) else o)))
            # the language model's input: the text embeddings with the image features scattered in
            hooks.append(lm.layers[0].register_forward_pre_hook(
                lambda m, a, kw: cap.__setitem__("embed", (a[0] if a else kw["hidden_states"]).clone()),
                with_kwargs=True))
            for li, layer in enumerate(lm.layers):
                hooks.append(layer.register_forward_hook(
                    lambda m, i, o, li=li: cap.__setitem__(li, o[0] if isinstance(o, tuple) else o)))
            t = torch.tensor([ids])
            last = [p for own in ends for p in own]
            rows = slot_logits(model, t, torch.ones_like(t), torch.tensor(last), torch.tensor([state_len]),
                               torch.tensor([question]), torch.zeros(len(last), dtype=torch.long), **inputs)
            for h in hooks:
                h.remove()
            codes = rows.codes.float()
            logz = codes - torch.logsumexp(codes, dim=-1, keepdim=True)

            if r == 0:
                np.save(os.path.join(out, "pixel_values.npy"), inputs["pixel_values"].float().numpy())
                np.save(os.path.join(out, "grid_thw.npy"), inputs["image_grid_thw"].numpy().astype(np.int64))
                np.save(os.path.join(out, "vis_embed.npy"), cap["vis_embed"].float().numpy())
                for li in range(len(tower.layers)):
                    np.save(os.path.join(out, f"vis_layer{li}.npy"), cap[f"vis{li}"].float().numpy())
                np.save(os.path.join(out, "vis_proj.npy"), cap["vis_proj"].float().numpy())
            np.save(os.path.join(out, f"row{r}_ids.npy"), np.array(ids, dtype=np.int32))
            np.save(os.path.join(out, f"row{r}_question.npy"), np.array(question, dtype=np.int32))
            np.save(os.path.join(out, f"row{r}_embed.npy"), cap["embed"][0].float().numpy())
            for li in range(len(lm.layers)):
                np.save(os.path.join(out, f"row{r}_layer{li}.npy"), cap[li][0].float().numpy())
            np.save(os.path.join(out, f"row{r}_final.npy"), cap["final"][0].float().numpy())
            np.save(os.path.join(out, f"row{r}_slot_logz.npy"), logz.numpy())
            with open(os.path.join(out, f"row{r}_meta.json"), "w") as f:
                json.dump({"first_question": k, "length": len(ids), "state_len": state_len, "slots": ends,
                           "grid_thw": inputs["image_grid_thw"].tolist()}, f, indent=1)
            print(f"  {name} row {r}: {len(ids)} tokens, state {state_len}, grids {inputs['image_grid_thw'].tolist()}, "
                  f"pixel_values {tuple(inputs['pixel_values'].shape)}, slots {ends}")
        print(f"  {name}: {json.dumps(answers['answers'], ensure_ascii=False)[:400]}")


if __name__ == "__main__":
    main()
