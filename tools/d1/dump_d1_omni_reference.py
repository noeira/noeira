#!/usr/bin/env python3
"""d1-Omni-2B's System One on a HEARD clip, run by its OWN code in float32, every
stage of the audio-input path dumped — the reference the Mojo port of the
audio path is gated against (docs/NN_MODEL_PORTS_PLAN.md, d1-Omni step 1).

    ~/Documents/noeira-parakeet/d1ref-venv/bin/python tools/d1/dump_d1_omni_reference.py \
        [--model ~/.cache/noeira/local-ai/d1/d1-Omni-2B] [--clips ~/.cache/noeira/parakeet/clips] \
        [--out ~/.cache/noeira/d1/omni_ref] [--cases heard_en,heard_fr,heard_neg] [--sweep]

⚠ RUN IT WHERE d1-Omni WAS WRITTEN FOR: transformers >= 5.19 (the Orin's
`d1ref-venv`: 5.19.0 with torch 2.8 on the CPU). Float32 on the CPU, ~8.7 GB of
weights. Nothing is written to the model's folder.

The path (`modeling_lfm3_omni.py`): 16 kHz waveform -> 128-bin log-mel
(`feature_extraction_lfm3_omni.py`) -> `Lfm3OmniDownsampler` (three stride-2
convs, one vector per 8 frames) -> a 17-layer LFM2 encoder (hidden 768, a
sliding-window causal mask of 32) -> `Lfm3OmniProjector` (LayerNorm, 768 -> 2048,
GELU, 2048 -> 2048) -> scattered into the prompt at the `<|audio|>` tokens ->
the LFM3 text backbone (the d1 one, its own weights) -> the slots.

Per case (`<out>/<case>/`):

    answers.json              `system_one`'s output, as a caller sees it
    prompt.json               the state prefix (the clip's markup at its head) and
                              each question's suffix (text), for the tokenizer gate
    wave.npy                  the waveform the processor took (16 kHz, float32) [N]
    mel.npy                   input_features                       [128, F]
    mel_mask.npy              feature_attention_mask               [F] int32
    down.npy                  the downsampler's output             [T_a, 768]
    enc_layer<l>.npy          encoder layer l's output, l = 0..16  [T_a, 768]
    enc_final.npy             the encoder's last hidden state      [T_a, 768]
    proj.npy                  the projector's output (the audio tokens' embeddings) [T_a, 2048]
    row<r>_ids.npy            the packed row's token ids (audio placeholders in) [T] int32
    row<r>_meta.json          state length, each question's slots, the audio span
    row<r>_question.npy       each token's question (1..N, -1 on the state) [T] int32
    row<r>_embed.npy          the text backbone's inputs_embeds (the audio scattered in) [T, 2048]
    row<r>_layer<l>.npy       decoder layer l's output              [T, 2048]
    row<r>_final.npy          after the final norm (`embedding_norm`) [T, 2048]
    row<r>_slot_logz.npy      log_softmax of the LM head at each slot [S, vocab]

The cases: a clip as the request's media and the room's questions (did the
speaker address the robot, where should it walk, how urgent) — in English
("walk to the fridge"), French ("va près du canapé") and an English negation
("don't go to the door"). `--sweep` also writes `<out>/sweep.json`: the same
questions over every clip of the manifest, answers only, each clip HEARD and
WRITTEN (its transcript as the state's "heard", no audio) — the step-3
comparison against Parakeet -> d1 reads it.
"""

import argparse
import importlib
import json
import os

import numpy as np
import torch
from transformers import AutoModel

ROOM_STATE = {
    "robot": {"at": "door", "facing": "kitchen", "doing": "standing"},
    "places": {"fridge": "3.1 m ahead, slightly left", "door": "here", "table": "2.0 m to the right",
               "sofa": "4.5 m behind"},
    "heard": "the attached recording",
    "recently": ["walked to the door"],
}

QUESTIONS = {
    "addressed": {"type": "noul", "instructions": "Is the speaker asking the robot to do something?"},
    "destination": {"type": "choice", "instructions": "Where should the robot walk?",
                    "criteria": {"fridge": "walk to the fridge", "door": "walk to the door",
                                 "table": "walk to the table", "sofa": "walk to the sofa",
                                 "nowhere": "do not walk anywhere"}},
    "urgency": {"type": "score", "instructions": "How urgent is the request?",
                "criteria": ["not urgent", "somewhat urgent", "very urgent"]},
}

CASES = {"heard_en": "en_13.wav", "heard_fr": "fr_05.wav", "heard_neg": "en_17.wav"}


def save(out, name, x):
    np.save(os.path.join(out, name), x.detach().float().numpy() if torch.is_tensor(x) else x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.expanduser("~/.cache/noeira/local-ai/d1/d1-Omni-2B"))
    ap.add_argument("--clips", default=os.path.expanduser("~/.cache/noeira/parakeet/clips"))
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/d1/omni_ref"))
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--written-only", action="store_true",
                    help="the sweep's written half only, no dumps: runs on d1-2B too (--model .../d1-2B)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=torch.float32)
    model.eval()
    eng = model.engine
    if args.written_only:
        rows = []
        with open(os.path.join(args.clips, "manifest.tsv")) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                w = model.system_one(dict(ROOM_STATE, heard=parts[3]), QUESTIONS)
                rows.append({"clip": parts[0], "lang": parts[1], "text": parts[3], "written": w["answers"]})
                dw = w["answers"]["destination"]
                print(f"  written {parts[0]} '{parts[3]}': {dw['choice']} {dw['confidence']:.2f}")
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"sweep_written_{os.path.basename(args.model.rstrip('/'))}.json"), "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        return
    core = model.model  # Lfm3OmniModel: language_model, audio_encoder, audio_projector
    lm = core.language_model
    enc = core.audio_encoder
    pkg = type(model).__module__.rsplit(".", 1)[0]
    slot_logits = importlib.import_module(pkg + ".slots").slot_logits
    as_media = importlib.import_module(pkg + ".media").as_media

    print(f"d1-Omni at {args.model}: attention {eng.attention}, set size {eng.set_size}; text "
          f"{len(lm.layers)} layers; audio encoder {len(enc.lfm.layers)} layers "
          f"{list(enc.lfm.config.layer_types)}, window {enc.config.sliding_window}")

    cap = {}

    def keep(key):
        return lambda m, i, o: cap.__setitem__(key, o[0] if isinstance(o, tuple) else o)

    hooks = [
        enc.register_forward_pre_hook(lambda m, a: cap.update(mel=a[0], mel_len=a[1])),
        enc.downsample.register_forward_hook(keep("down")),
        enc.lfm.register_forward_hook(lambda m, i, o: cap.__setitem__("enc_final", o.last_hidden_state)),
        core.audio_projector.register_forward_hook(keep("proj")),
        lm.register_forward_pre_hook(lambda m, a, kw: cap.__setitem__("embed", kw["inputs_embeds"]),
                                     with_kwargs=True),
        lm.embedding_norm.register_forward_hook(keep("final")),
    ]
    for li, layer in enumerate(enc.lfm.layers):
        hooks.append(layer.register_forward_hook(keep(("enc", li))))
    for li, layer in enumerate(lm.layers):
        hooks.append(layer.register_forward_hook(keep(("lm", li))))

    for name in args.cases.split(","):
        clip = os.path.join(args.clips, CASES[name])
        media = [{"type": "audio", "path": clip}]
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)
        answers = model.system_one(ROOM_STATE, QUESTIONS, media=media)
        with open(os.path.join(out, "answers.json"), "w") as f:
            json.dump(answers, f, ensure_ascii=False, indent=1)

        qs = [eng.ask(q) for q in QUESTIONS.values()]
        items = [as_media(m) for m in media]
        prefix, _ = eng._prefix(ROOM_STATE, items)
        suffixes, offsets, _ = eng._suffixes(qs)
        with open(os.path.join(out, "prompt.json"), "w") as f:
            json.dump({"prefix": prefix, "suffixes": suffixes, "slot_offsets": offsets, "names": list(QUESTIONS),
                       "clip": CASES[name]}, f, ensure_ascii=False, indent=1)
        np.save(os.path.join(out, "wave.npy"), np.asarray(eng.media._decoded(items[0]), dtype=np.float32))

        seqs = eng._sequences([(ROOM_STATE, qs, media)])
        for r, (_, k, ids, state_len, question, ends, inputs) in enumerate(seqs):
            cap.clear()
            t = torch.tensor([ids])
            last = [p for own in ends for p in own]
            rows = slot_logits(model, t, torch.ones_like(t), torch.tensor(last), torch.tensor([state_len]),
                               torch.tensor([question]), torch.zeros(len(last), dtype=torch.long), **inputs)
            codes = rows.codes.float()
            logz = codes - torch.logsumexp(codes, dim=-1, keepdim=True)
            audio = [p for p, i in enumerate(ids) if i == core.config.audio_token_id]

            if r == 0:  # the clip heads the state: every row holds the same audio
                save(out, "mel.npy", cap["mel"][0])
                save(out, "mel_mask.npy", inputs["feature_attention_mask"][0].to(torch.int32).numpy())
                save(out, "down.npy", cap["down"][0].transpose(0, 1))
                for li in range(len(enc.lfm.layers)):
                    save(out, f"enc_layer{li}.npy", cap[("enc", li)][0])
                save(out, "enc_final.npy", cap["enc_final"][0])
                save(out, "proj.npy", cap["proj"])
            np.save(os.path.join(out, f"row{r}_ids.npy"), np.array(ids, dtype=np.int32))
            np.save(os.path.join(out, f"row{r}_question.npy"), np.array(question, dtype=np.int32))
            save(out, f"row{r}_embed.npy", cap["embed"][0])
            for li in range(len(lm.layers)):
                save(out, f"row{r}_layer{li}.npy", cap[("lm", li)][0])
            save(out, f"row{r}_final.npy", cap["final"][0])
            save(out, f"row{r}_slot_logz.npy", logz)
            with open(os.path.join(out, f"row{r}_meta.json"), "w") as f:
                json.dump({"first_question": k, "length": len(ids), "state_len": state_len, "slots": ends,
                           "audio_first": audio[0] if audio else -1, "audio_tokens": len(audio),
                           "mel_frames": int(inputs["feature_attention_mask"].sum())}, f, indent=1)
            print(f"  {name} row {r}: {len(ids)} tokens ({len(audio)} audio from {audio[0] if audio else -1}), "
                  f"state {state_len}, {int(inputs['feature_attention_mask'].sum())} mel frames, slots {ends}")
        print(f"  {name} ({CASES[name]}): {json.dumps(answers['answers'], ensure_ascii=False)[:400]}")

    if args.sweep:
        for h in hooks:
            h.remove()
        rows = []
        with open(os.path.join(args.clips, "manifest.tsv")) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                a = model.system_one(ROOM_STATE, QUESTIONS, media=[{"type": "audio",
                                                                    "path": os.path.join(args.clips, parts[0])}])
                # the same model and questions with the TRANSCRIPT as text: what hearing costs, apart from the state
                w = model.system_one(dict(ROOM_STATE, heard=parts[3]), QUESTIONS)
                rows.append({"clip": parts[0], "lang": parts[1], "text": parts[3], "answers": a["answers"],
                             "written": w["answers"]})
                d, dw = a["answers"]["destination"], w["answers"]["destination"]
                print(f"  sweep {parts[0]} '{parts[3]}': heard {d['choice']} {d['confidence']:.2f}, "
                      f"written {dw['choice']} {dw['confidence']:.2f}")
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "sweep.json"), "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
