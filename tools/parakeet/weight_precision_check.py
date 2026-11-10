#!/usr/bin/env python3
"""Do the reference transcripts survive lower-precision GEMM weights?

    pixi run -e detect python tools/parakeet/weight_precision_check.py \
        --manifest ~/.cache/noeira/parakeet/clips/manifest.tsv

The question to answer BEFORE building a half-precision path in the port
(docs/NN_MODEL_PORTS_PLAN.md §10.6): on the Orin the encoder's GEMM time is the
streaming of its fp32 weights, so halving the weights should halve it — if the
model's output does not move.

For each precision, the weights of the encoder blocks' GEMMs (the two
feed-forwards, the five attention projections, the two pointwise convolutions:
264 tensors, 96 % of the parameters) are rounded to that precision and back to
float32; everything else — activations, norms, biases, the depthwise
convolutions, the decoder — stays float32, which is what the port would do.
Every clip is then transcribed as cut and with 0.5 s of silence on both sides,
and compared with the unmodified model's transcript of the same form.

It reports, per precision: transcripts changed (of 2 x clips), and the largest
change of the encoder output relative to its largest value.

Measured 2026-11-10 on the 18 clips: float16 changes 0 of 36 transcripts and
moves the encoder output by at most 1.2e-2; bfloat16 changes 1 of 36 ("Who to
the door?" -> "Go to the door!") and moves it by 0.18. Hence float16.
"""

import argparse
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, ParakeetForTDT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_parakeet_reference import GEMM_WEIGHTS as GEMM, MODEL  # noqa: E402  (one list, in one place)


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def run(model, proc, clips):
    """[(text, encoder output)] for each clip in both forms."""
    out = []
    for audio in clips:
        for pad in (0.0, 0.5):
            a = audio
            if pad > 0:
                z = np.zeros(int(pad * 16000), dtype=np.float32)
                a = np.concatenate([z, audio, z])
            inputs = proc(a, sampling_rate=16000, return_tensors="pt")
            enc = model.get_audio_features(**inputs).last_hidden_state[0].numpy()
            gen = model.generate(**inputs)
            out.append((proc.batch_decode(gen.sequences, skip_special_tokens=True)[0], enc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--precisions", default="float16,bfloat16")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = ParakeetForTDT.from_pretrained(MODEL, dtype=torch.float32).eval()
    man = Path(args.manifest)
    rows = [l.split("\t") for l in man.read_text().splitlines() if l.strip()]
    clips = [read_wav(man.parent / r[0]) for r in rows]

    sd = model.state_dict()
    names = [k for k in sd if GEMM.match(k)]
    n_gemm = sum(sd[k].numel() for k in names)
    n_all = sum(v.numel() for k, v in sd.items() if v.is_floating_point())
    print(f"{len(names)} GEMM weight tensors, {n_gemm / 1e6:.1f} M of {n_all / 1e6:.1f} M parameters"
          f" ({100 * n_gemm / n_all:.1f} %), largest |w| = {max(float(sd[k].abs().max()) for k in names):.2f}")
    saved = {k: sd[k].clone() for k in names}

    base = run(model, proc, clips)
    for prec in args.precisions.split(","):
        dt = getattr(torch, prec)
        worst_w = 0.0
        for k in names:
            q = saved[k].to(dt).to(torch.float32)
            worst_w = max(worst_w, float(((q - saved[k]).abs() / saved[k].abs().clamp_min(1e-12)).median()))
            sd[k].copy_(q)
        got = run(model, proc, clips)
        changed = [(i, b[0], g[0]) for i, (b, g) in enumerate(zip(base, got)) if b[0] != g[0]]
        rel = max(float(np.abs(g[1] - b[1]).max() / np.abs(b[1]).max()) for b, g in zip(base, got))
        print(f"{prec}: {len(changed)} of {len(base)} transcripts changed; encoder output moved by at most"
              f" {rel:.2e} (relative); median weight rounding error {worst_w:.1e}")
        for i, b, g in changed:
            print(f"   {rows[i // 2][0]} {'padded' if i % 2 else 'as cut'}: '{b}' -> '{g}'")
        for k in names:
            sd[k].copy_(saved[k])


if __name__ == "__main__":
    main()
