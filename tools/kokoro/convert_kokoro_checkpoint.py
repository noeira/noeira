#!/usr/bin/env python3
"""Kokoro's published checkpoint and voices, as safetensors the Mojo port reads.

    pixi run -e detect python tools/kokoro/convert_kokoro_checkpoint.py \
        [--out ~/.cache/noeira/kokoro] [--voices af_heart,ff_siwis,...]

`kokoro-v1_0.pth` is a torch pickle of five state dicts (`bert`, `bert_encoder`,
`predictor`, `decoder`, `text_encoder`) whose keys carry a `module.` prefix.
This writes them as ONE flat safetensors file, `kokoro-v1_0.safetensors`, keys
`<group>.<name without module.>`, values untouched: weight-normalised
convolutions stay as their `weight_g` / `weight_v` pair (the port folds them at
load, so a fine-tune can keep the parametrisation).

A voice is a `[510, 1, 256]` table: row `n - 1` is the style for an utterance
of `n` phonemes (decoder style in columns 0-127, prosody style in 128-255).
Each becomes `voices/<name>.safetensors`, key `pack`, shape `[510, 256]`.
`config.json` (which holds the phoneme vocabulary) is copied beside them.

A conversion, not inference: the weights are the published ones, byte for byte
(it prints the tensor count and the parameter total to check against the plan).

It also writes `kokoro-v1_0.check.tsv`, the reference of the Mojo loader's gate
(`tests/audio/test_kokoro_weights.mojo`): one line per tensor AS THE PORT HOLDS
IT — a weight-normalised pair folded by torch's own `_weight_norm` into
`<prefix>.weight` — with its shape and a position-weighted float64 sum
(`sum(w[k] * (1 + k % 7))`, which a transposed, shifted or unfolded tensor
does not reproduce). For a folded tensor a second sum, of the UNFOLDED
`weight_v`, is given: the gate checks it would have told the two apart.
"""

import argparse
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reference import snapshot_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/.cache/noeira/kokoro"))
    ap.add_argument("--voices", default=None, help="comma list; default: every voice in the snapshot")
    args = ap.parse_args()
    snap = snapshot_dir()
    os.makedirs(os.path.join(args.out, "voices"), exist_ok=True)

    groups = torch.load(snap + "kokoro-v1_0.pth", map_location="cpu", weights_only=True)
    flat = {}
    for group, state in groups.items():
        for key, value in state.items():
            name = key[len("module."):] if key.startswith("module.") else key
            flat[f"{group}.{name}"] = value.contiguous()
    n_float = sum(v.numel() for v in flat.values() if v.is_floating_point())
    save_file(flat, os.path.join(args.out, "kokoro-v1_0.safetensors"))
    print(f"{len(flat)} tensors, {n_float / 1e6:.2f} M float parameters -> {args.out}/kokoro-v1_0.safetensors")
    def wsum(t):
        a = t.double().reshape(-1)
        return float((a * (1 + torch.arange(a.numel(), dtype=torch.float64) % 7)).sum())

    lines = []
    for key in sorted(flat):
        if key.endswith(".weight_g"):
            continue
        if key.endswith(".weight_v"):
            prefix = key[: -len(".weight_v")]
            w = torch._weight_norm(flat[key], flat[prefix + ".weight_g"], 0)
            lines.append((prefix + ".weight", w, wsum(flat[key])))
        else:
            lines.append((key, flat[key], None))
    with open(os.path.join(args.out, "kokoro-v1_0.check.tsv"), "w") as f:
        for name, t, v_sum in lines:
            shape = ",".join(str(d) for d in t.shape)
            f.write(f"{name}\t{shape}\t{wsum(t)!r}\t{'' if v_sum is None else repr(v_sum)}\n")
    print(f"{len(lines)} tensors as the port holds them -> {args.out}/kokoro-v1_0.check.tsv")
    other = sorted({str(v.dtype) for v in flat.values() if not v.is_floating_point()})
    if other:
        print("  non-float tensors:", other, [k for k, v in flat.items() if not v.is_floating_point()][:5])

    # The phoneme vocabulary lives in the repository's config.json.
    shutil.copyfile(snap + "config.json", os.path.join(args.out, "config.json"))

    names = args.voices.split(",") if args.voices else sorted(
        f[:-3] for f in os.listdir(snap + "voices") if f.endswith(".pt")
    )
    for name in names:
        pack = torch.load(snap + f"voices/{name}.pt", map_location="cpu", weights_only=True)
        assert tuple(pack.shape) == (510, 1, 256), (name, pack.shape)
        save_file({"pack": pack[:, 0, :].contiguous()}, os.path.join(args.out, "voices", f"{name}.safetensors"))
    print(f"{len(names)} voices -> {args.out}/voices/")


if __name__ == "__main__":
    main()
