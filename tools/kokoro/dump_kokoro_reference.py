#!/usr/bin/env python3
"""Dump Kokoro-82M's stages from hexgrad's own model code, for the Mojo port.

    pixi run -e detect python tools/kokoro/dump_kokoro_reference.py \
        --cases ~/.cache/noeira/kokoro/cases.json --out ~/.cache/noeira/kokoro/ref

`cases.json` is a list of `{name, voice, phonemes, ...}`: the model is gated on
PHONEME strings, so the text front end is not part of this reference
(docs/NN_MODEL_PORTS_PLAN.md §11). For each case, `<out>/<name>/` gets, as
`.npy` (float32 unless noted):

    input_ids   [T] int64        0, the phoneme ids, 0
    ref_s       [256]            the voice's style row for this length
    bert        [T, 768]         ALBERT's last hidden state
    d_en        [512, T]         bert_encoder(bert), channels first
    dur_enc     [T, 640]         predictor.text_encoder (duration encoder)
    dur_lstm    [T, 512]         predictor.lstm
    duration    [T]              sum of the 50 sigmoids (before rounding)
    pred_dur    [T] int64        frames per phoneme: the alignment
    en          [640, F]         dur_enc expanded to F = sum(pred_dur) frames
    f0, n       [2F]             predicted pitch and energy curves
    t_en        [512, T]         text_encoder
    asr         [512, F]         t_en expanded
    dec_encode, dec_decode_0..3  the decoder's AdaIN blocks
    gen_source  [600 F]          the harmonic source (tanh of the merged sines)
    gen_har     [22, 120F+1]     STFT (magnitude | phase) of the harmonic source
    gen_ups_0, gen_ups_1         after each transposed convolution
    gen_res_0_lrelu, gen_res_1_lrelu   leaky_relu of each stage's averaged residual
                                 blocks (what the next conv reads)
    gen_post    [22, frames]     conv_post: log-magnitude | phase argument
    audio       [F * 600]        24 kHz

⚠ THE SOURCE MODULE DRAWS RANDOM NUMBERS (a random initial phase per harmonic,
Gaussian noise on the sines, a noise source). They are recorded in call order
as `noise_0.npy`, `noise_1.npy`, ... and the port must be fed the same values:
only then is the source comparable at all. The seed is fixed, so two runs of
this tool write the same files.

⚠ AND EVEN THEN THE SOURCE IS ONLY REPRODUCIBLE TO ~1e-3. It is `sin` of a phase
accumulated in float32 over the whole utterance — thousands of radians, where
one float32 step is ~5e-4 — so the reference carries its own rounding there: a
float64 evaluation of the same formulas differs from it by 9e-4 (peak 0.15), a
float32 one in another operation order by 1.6e-3. The vocoder then amplifies
any source difference (its output passes through an `exp`). So the port's
vocoder is gated FROM `gen_har`, the reference's own source spectrum, where it
must match tightly, and the source is gated separately, within that noise.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reference import load_kmodel  # noqa: E402


class NoiseTap:
    """Records every tensor `torch.rand` / `torch.randn_like` returns while active."""

    def __init__(self):
        self.drawn = []

    def __enter__(self):
        self._rand, self._randn_like = torch.rand, torch.randn_like

        def rand(*a, **k):
            t = self._rand(*a, **k)
            self.drawn.append(("rand", t.detach().clone()))
            return t

        def randn_like(*a, **k):
            t = self._randn_like(*a, **k)
            self.drawn.append(("randn_like", t.detach().clone()))
            return t

        torch.rand, torch.randn_like = rand, randn_like
        return self

    def __exit__(self, *exc):
        torch.rand, torch.randn_like = self._rand, self._randn_like


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.expanduser("~/.cache/noeira/kokoro/cases.json"))
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/kokoro/ref"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    model, snap = load_kmodel()
    cases = json.load(open(args.cases))

    taps = {}

    def hook(name):
        def fn(_module, _inp, out):
            taps[name] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return fn

    def in_hook(name):
        def fn(_module, inp):
            taps[name] = inp[0].detach().clone()
        return fn

    dec, gen, pred = model.decoder, model.decoder.generator, model.predictor
    hooks = [
        model.bert.register_forward_hook(hook("bert")),
        model.bert_encoder.register_forward_hook(hook("bert_encoder")),
        pred.text_encoder.register_forward_hook(hook("dur_enc")),
        pred.lstm.register_forward_hook(hook("dur_lstm")),
        model.text_encoder.register_forward_hook(hook("t_en")),
        dec.encode.register_forward_hook(hook("dec_encode")),
        gen.noise_convs[0].register_forward_pre_hook(in_hook("gen_har")),
        gen.m_source.register_forward_hook(hook("gen_source")),
        gen.conv_post.register_forward_hook(hook("gen_post")),
    ]
    hooks += [dec.decode[i].register_forward_hook(hook(f"dec_decode_{i}")) for i in range(len(dec.decode))]
    hooks += [gen.ups[i].register_forward_hook(hook(f"gen_ups_{i}")) for i in range(len(gen.ups))]
    # The input of stage i+1's transposed conv is leaky_relu(stage i's output);
    # conv_post's input is leaky_relu(the last stage's output).
    hooks.append(gen.ups[1].register_forward_pre_hook(in_hook("gen_res_0_lrelu")))
    hooks.append(gen.conv_post.register_forward_pre_hook(in_hook("gen_res_1_lrelu")))

    for c in cases:
        ps = c["phonemes"]
        ids = [model.vocab.get(p) for p in ps]
        if any(i is None for i in ids):
            raise SystemExit(f"{c['name']}: phoneme(s) outside the vocabulary: {[p for p, i in zip(ps, ids) if i is None]}")
        input_ids = torch.LongTensor([[0, *ids, 0]])
        pack = torch.load(snap + f"voices/{c['voice']}.pt", map_location="cpu", weights_only=True)
        ref_s = pack[len(ps) - 1]
        taps.clear()
        torch.manual_seed(args.seed)
        with NoiseTap() as noise:
            audio, pred_dur = model.forward_with_tokens(input_ids, ref_s, 1.0)

        # Recompute the few values the model's forward keeps local.
        s = ref_s[:, 128:]
        d = taps["dur_enc"]
        duration = torch.sigmoid(pred.duration_proj(taps["dur_lstm"])).sum(axis=-1)[0]
        t = input_ids.shape[1]
        indices = torch.repeat_interleave(torch.arange(t), pred_dur)
        aln = torch.zeros((t, indices.shape[0]))
        aln[indices, torch.arange(indices.shape[0])] = 1
        en = d.transpose(-1, -2) @ aln.unsqueeze(0)
        with NoiseTap():
            f0, n = pred.F0Ntrain(en, s)
        asr = taps["t_en"] @ aln.unsqueeze(0)
        assert torch.equal(torch.round(duration).clamp(min=1).long(), pred_dur), "duration recomputation drifted"

        out = os.path.join(args.out, c["name"])
        os.makedirs(out, exist_ok=True)

        def save(name, arr, dtype=np.float32):
            np.save(os.path.join(out, name + ".npy"), np.ascontiguousarray(np.asarray(arr), dtype=dtype))

        save("input_ids", input_ids[0].numpy(), np.int64)
        save("ref_s", ref_s[0].numpy())
        save("bert", taps["bert"][0].numpy())
        save("d_en", taps["bert_encoder"][0].numpy().T)
        save("dur_enc", d[0].numpy())
        save("dur_lstm", taps["dur_lstm"][0].numpy())
        save("duration", duration.numpy())
        save("pred_dur", pred_dur.numpy(), np.int64)
        save("en", en[0].numpy())
        save("f0", f0[0].numpy())
        save("n", n[0].numpy())
        save("t_en", taps["t_en"][0].numpy())
        save("asr", asr[0].numpy())
        save("dec_encode", taps["dec_encode"][0].numpy())
        for i in range(len(dec.decode)):
            save(f"dec_decode_{i}", taps[f"dec_decode_{i}"][0].numpy())
        save("gen_source", taps["gen_source"][0, :, 0].numpy())
        save("gen_har", taps["gen_har"][0].numpy())
        for i in range(len(gen.ups)):
            save(f"gen_ups_{i}", taps[f"gen_ups_{i}"][0].numpy())
        save("gen_res_0_lrelu", taps["gen_res_0_lrelu"][0].numpy())
        save("gen_res_1_lrelu", taps["gen_res_1_lrelu"][0].numpy())
        save("gen_post", taps["gen_post"][0].numpy())
        save("audio", audio.numpy())
        for i, (kind, t_) in enumerate(noise.drawn):
            save(f"noise_{i}", t_.numpy())
        with open(os.path.join(out, "case.json"), "w") as f:
            json.dump(
                {**c, "tokens": int(t), "frames": int(pred_dur.sum()), "samples": int(audio.numel()),
                 "noise": [[k, list(t_.shape)] for k, t_ in noise.drawn]},
                f, ensure_ascii=False, indent=1,
            )
        print(f"{out}: {t} tokens, {int(pred_dur.sum())} frames, {audio.numel()} samples,"
              f" noise draws {[(k, tuple(t_.shape)) for k, t_ in noise.drawn]}")
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
