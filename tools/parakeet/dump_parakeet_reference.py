#!/usr/bin/env python3
"""Dump Parakeet-TDT-0.6B-v3 activations from transformers' OWN implementation.

    pixi run -e detect python tools/parakeet/dump_parakeet_reference.py \
        --wav clip.wav --out /tmp/parakeet_ref [--pad-s 8]

The reference for the Mojo port (docs/NN_MODEL_PORTS_PLAN.md, M1). It drives
`ParakeetForTDT`, `ParakeetFeatureExtractor` and the tokenizer as shipped, on the
ORIGINAL nvidia weights, CPU fp32, and writes every stage the port has to
reproduce as `.npy`:

    audio.npy               [n]            the waveform fed in, float32
    mel_filters.npy         [128, 257]     librosa's slaney filterbank (the extractor's own)
    window.npy              [400]          hann, periodic=False
    features.npy            [frames, 128]  normalized log-mel, zero beyond the valid length
    feature_len.npy         [1]            valid mel frames
    subsampling.npy         [T, 1024]      after the conv subsampling + linear
    pos_embed.npy           [2T-1, 1024]   relative positional encoding
    layer_XX.npy            [T, 1024]      output of encoder block XX (00..23)
    encoder_proj.npy        [T, 640]       encoder_projector(last hidden state)
    encoder_len.npy         [1]            valid encoder frames
    tokens.npy, durations.npy  [steps]     the greedy TDT trace (blank included)
    transcript.txt

`--pad-s S` runs the SAME clip through the reference's own padded path: batched
with S seconds of silence, so the clip's row is zero-padded to S seconds and
masked. That is the fixed-maximum shape the Mojo port compiles for; arrays are
then written at the PADDED sizes and `feature_len` / `encoder_len` give the valid
part. `--compare-pad a,b,c` prints, for each padded length, the largest
difference on the valid frames against the unpadded run, whether the token trace
is the same, and the encoder time.

`--weights float16` rounds the encoder blocks' GEMM weights (`GEMM_WEIGHTS`:
the two feed-forwards, the five attention projections, the two pointwise
convolutions — 264 tensors, 96 % of the parameters) to that precision and back
to float32 before running, everything else untouched. That is the reference of
the port's half-precision build (`ParakeetSTT[..., WEIGHTS="fp16"]`), which
stores exactly those weights as float16: gated against the float32 reference
it would be "wrong" by the rounding itself (1e-2 on the encoder output), and a
bound that loose would pass real defects.

`ParakeetFeatureExtractor` takes its mel filterbank from `librosa`, which is why
the `detect` environment lists it (pixi.toml).
"""

import argparse
import os
import re
import time
import wave

import numpy as np

MODEL = "nvidia/parakeet-tdt-0.6b-v3"
GEMM_WEIGHTS = re.compile(
    r"^encoder\.layers\.\d+\.(feed_forward[12]\.linear[12]|self_attn\.(q|k|v|o|relative_k)_proj"
    r"|conv\.pointwise_conv[12])\.weight$"
)
"""The weights the port's half-precision build stores as float16 (its
`pk_gemm_slot`): keep the two lists the same."""


def round_gemm_weights(model, precision, torch):
    """Round the blocks' GEMM weights to `precision` and back, in place.
    Returns the tensors' names."""
    sd = model.state_dict()
    names = [k for k in sd if GEMM_WEIGHTS.match(k)]
    if len(names) != 11 * model.config.encoder_config.num_hidden_layers:
        raise SystemExit(f"expected 11 GEMM weights per block, matched {len(names)}")
    dt = getattr(torch, precision)
    for k in names:
        sd[k].copy_(sd[k].to(dt).to(torch.float32))
    return names


def read_wav(path):
    with wave.open(path, "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            raise SystemExit(f"{path}: need 16 kHz mono 16-bit PCM")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def run(model, proc, audio, pad_s, torch):
    """One clip through the reference. `pad_s > 0`: batched with silence of that
    length, so the clip is padded by the extractor and masked by the model."""
    if pad_s > 0:
        n_pad = int(round(pad_s * 16000))
        if n_pad < len(audio):
            raise SystemExit(f"--pad-s {pad_s} is shorter than the clip")
        batch = [audio, np.zeros(n_pad, dtype=np.float32)]
    else:
        batch = [audio]
    inputs = proc(batch, sampling_rate=16000, return_tensors="pt")
    feats, mask = inputs["input_features"][:1], inputs["attention_mask"][:1]
    t0 = time.perf_counter()
    enc = model.get_audio_features(input_features=feats, attention_mask=mask, output_hidden_states=True)
    enc_ms = (time.perf_counter() - t0) * 1e3
    gen = model.generate(input_features=feats, attention_mask=mask)
    text = proc.batch_decode(gen.sequences, skip_special_tokens=True)[0]
    return {
        "features": feats[0].numpy(),
        "feature_len": int(mask[0].sum()),
        "hidden": [h[0].numpy() for h in enc.hidden_states],
        "last": enc.last_hidden_state[0].numpy(),
        "proj": enc.pooler_output[0].numpy(),
        "encoder_len": int(enc.attention_mask[0].sum()),
        "tokens": gen.sequences[0].numpy(),
        "durations": gen.durations[0].numpy(),
        "text": text,
        "enc_ms": enc_ms,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pad-s", type=float, default=0.0)
    ap.add_argument("--compare-pad", default=None, help="comma list of padded lengths in seconds")
    ap.add_argument("--weights", default="float32", choices=["float32", "float16", "bfloat16"],
                    help="precision the blocks' GEMM weights are rounded to")
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, ParakeetForTDT

    torch.set_grad_enabled(False)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = ParakeetForTDT.from_pretrained(MODEL, dtype=torch.float32).eval()
    if args.weights != "float32":
        n = len(round_gemm_weights(model, args.weights, torch))
        print(f"{n} GEMM weight tensors rounded to {args.weights}")
    audio = read_wav(args.wav)

    if args.compare_pad:
        base = run(model, proc, audio, 0.0, torch)
        base = run(model, proc, audio, 0.0, torch)  # second run: the first pays one-time setup
        t = base["encoder_len"]
        print(f"unpadded  {len(audio) / 16000:.2f} s  T={t}  encoder {base['enc_ms']:.0f} ms  '{base['text']}'")
        for s in [float(x) for x in args.compare_pad.split(",")]:
            r = run(model, proc, audio, s, torch)
            assert r["encoder_len"] == t and r["feature_len"] == base["feature_len"]
            d_feat = np.abs(r["features"][: base["feature_len"]] - base["features"][: base["feature_len"]]).max()
            d_enc = np.abs(r["last"][:t] - base["last"][:t]).max()
            ref = np.abs(base["last"][:t]).max()
            same = np.array_equal(r["tokens"], base["tokens"]) and np.array_equal(r["durations"], base["durations"])
            print(
                f"pad {s:5.1f} s  T_max={r['last'].shape[0]:4d}  encoder {r['enc_ms']:6.0f} ms"
                f"  max|d features| {d_feat:.2e}  max|d encoder| {d_enc:.2e} (scale {ref:.1f})"
                f"  trace {'SAME' if same else 'DIFFERENT'}  '{r['text']}'"
            )
        return

    if not args.out:
        raise SystemExit("--out is required without --compare-pad")
    r = run(model, proc, audio, args.pad_s, torch)
    os.makedirs(args.out, exist_ok=True)

    def save(name, arr):
        np.save(os.path.join(args.out, name + ".npy"), np.ascontiguousarray(arr))

    fe = proc.feature_extractor
    save("audio", audio)
    save("mel_filters", fe.mel_filters.numpy().astype(np.float32))
    save("window", torch.hann_window(fe.win_length, periodic=False).numpy())
    save("features", r["features"].astype(np.float32))
    save("feature_len", np.array([r["feature_len"]], dtype=np.int64))
    hs = r["hidden"]
    # `hidden_states` records the encoder blocks; whether the subsampling output
    # leads the tuple depends on the transformers version, so it is checked.
    n_layers = model.config.encoder_config.num_hidden_layers
    if len(hs) == n_layers + 1:
        save("subsampling", hs[0].astype(np.float32))
        hs = hs[1:]
    assert len(hs) == n_layers, len(hs)
    assert np.array_equal(hs[-1], r["last"])
    for i, h in enumerate(hs):
        save(f"layer_{i:02d}", h.astype(np.float32))
    pos = model.encoder.encode_positions(torch.zeros(1, r["last"].shape[0], 1))[0].numpy()
    save("pos_embed", pos.astype(np.float32))
    save("encoder_proj", r["proj"].astype(np.float32))
    save("encoder_len", np.array([r["encoder_len"]], dtype=np.int64))
    save("tokens", r["tokens"].astype(np.int64))
    save("durations", r["durations"].astype(np.int64))
    with open(os.path.join(args.out, "transcript.txt"), "w") as f:
        f.write(r["text"] + "\n")
    print(
        f"{args.out}: {len(audio)} samples, {r['feature_len']} mel frames of {r['features'].shape[0]},"
        f" {r['encoder_len']} encoder frames of {r['last'].shape[0]}, '{r['text']}'"
    )


if __name__ == "__main__":
    main()
