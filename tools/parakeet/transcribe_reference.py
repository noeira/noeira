#!/usr/bin/env python3
"""Transcribe a clip manifest with the REFERENCE Parakeet-TDT (transformers, CPU fp32).

    .pixi/envs/detect/bin/python tools/parakeet/transcribe_reference.py \
        --manifest ~/.cache/noeira/parakeet/clips/manifest.tsv [--pad-s 0.5] [--out ref.tsv]

`manifest.tsv`: `file <TAB> lang <TAB> voice <TAB> expected text`, 16 kHz mono
16-bit WAVs beside it. The transcripts are the end-to-end metric the Mojo port
is held to (docs/NN_MODEL_PORTS_PLAN.md, M1). `--pad-s` adds silence on both
sides of each clip: a short command cut tight is transcribed worse (§10 there).
"""
import argparse
import time
import wave
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
ap.add_argument("--out", default=None)
ap.add_argument("--pad-s", type=float, default=0.0, help="silence added on both sides")
args = ap.parse_args()

import torch  # noqa: E402
from transformers import AutoProcessor, ParakeetForTDT  # noqa: E402

torch.set_grad_enabled(False)
proc = AutoProcessor.from_pretrained(args.model)
model = ParakeetForTDT.from_pretrained(args.model, dtype=torch.float32).eval()
print("params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


man = Path(args.manifest)
rows = [l.rstrip("\n").split("\t") for l in man.read_text().splitlines() if l.strip()]
lines = []
for f, lang, voice, text in rows:
    audio = read_wav(man.parent / f)
    if args.pad_s > 0:
        z = np.zeros(int(args.pad_s * 16000), dtype=np.float32)
        audio = np.concatenate([z, audio, z])
    t0 = time.perf_counter()
    inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
    out = model.generate(**inputs)
    hyp = proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
    dt = (time.perf_counter() - t0) * 1e3
    line = f"{f}\t{lang}\t{len(audio) / 16000:.2f}s\t{dt:.0f}ms\t{text}\t=>\t{hyp}"
    print(line, flush=True)
    lines.append(line)
if args.out:
    Path(args.out).write_text("\n".join(lines) + "\n")
