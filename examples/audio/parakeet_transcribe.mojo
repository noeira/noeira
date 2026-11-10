# +--------------------------------------------------------------------------+ #
# | Speech-to-text on this machine: Parakeet-TDT in Mojo
# +--------------------------------------------------------------------------+ #
"""Transcribe WAV files with `nvidia/parakeet-tdt-0.6b-v3`, no Python, no server.

    pixi run mojo build -I . examples/audio/parakeet_transcribe.mojo -o build/parakeet_transcribe
    ./build/parakeet_transcribe [--pad 0.5] [--profile] clip.wav [more.wav ...]

Needs the checkpoint in the Hub cache (2.5 GB):

    hf download nvidia/parakeet-tdt-0.6b-v3 model.safetensors tokenizer.json config.json

`--pad S` adds S seconds of silence on both sides (default 0.5: the model hears
a command cut tight markedly worse). Any sample rate and channel count is
accepted; the clip is mixed to mono and resampled to 16 kHz. `--profile`
splits the time of each clip into its stages.

CPU, fp32, compiled for clips of up to 8 s.
"""

from std.sys import argv
from std.time import perf_counter_ns

from noeira.io.wav import load_wav
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.audio.parakeet.config import PK_LAYERS
from noeira.audio.parakeet.decode import pk_greedy_decode
from noeira.audio.parakeet.encoder import PkEncoderWork, pk_subsample, pk_block
from noeira.audio.parakeet.stt import ParakeetSTT


comptime F_MAX: Int = 801


def ms(a: Int, b: Int) -> Float64:
    return Float64(b - a) / 1.0e6


def main() raises:
    var args = argv()
    var pad_s = 0.5
    var profile = False
    var files = List[String]()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--pad":
            i += 1
            pad_s = atof(args[i])
        elif a == "--profile":
            profile = True
        else:
            files.append(a^)
        i += 1
    if len(files) == 0:
        print("usage: parakeet_transcribe [--pad 0.5] [--profile] clip.wav [more.wav ...]")
        return

    var t0 = perf_counter_ns()
    var stt = ParakeetSTT[F_MAX].load()
    print("loaded in", Int(ms(t0, perf_counter_ns())), "ms; compiled for",
          ParakeetSTT[F_MAX].max_seconds(), "s")

    for f in range(len(files)):
        var wav = load_wav(files[f]).to_mono().resample(16000)
        var audio = wav.to_float32()
        if not profile:
            var a0 = perf_counter_ns()
            var r = stt.transcribe(audio, pad_s)
            print(files[f], " ", Int(ms(a0, perf_counter_ns())), "ms  ", r.text)
            continue

        # The same steps `transcribe` runs, timed one by one.
        var pad = Int(pad_s * 16000.0)
        var padded = List[Float32](length=len(audio) + 2 * pad, fill=0.0)
        for k in range(len(audio)):
            padded[pad + k] = audio[k]
        var p0 = perf_counter_ns()
        var n_mel = stt.front.extract[F_MAX](padded, stt.enc.feats.data)
        var p1 = perf_counter_ns()
        var valid = pk_subsample[F_MAX](stt.model, stt.enc, n_mel)
        var p2 = perf_counter_ns()
        for layer in range(PK_LAYERS):
            pk_block[F_MAX](stt.model, stt.enc, layer, valid)
        var p3 = perf_counter_ns()
        stt.model.enc_proj.forward["cpu", PkEncoderWork[F_MAX].T](
            TensorRefs[1](stt.enc.x), stt.enc.proj
        )
        var trace = pk_greedy_decode(
            stt.model, stt.enc.proj, valid, PkEncoderWork[F_MAX].T, stt.dec
        )
        var p4 = perf_counter_ns()
        # The GEMMs of the 24 blocks alone, on whatever the buffers hold.
        comptime T = PkEncoderWork[F_MAX].T
        comptime P = PkEncoderWork[F_MAX].P
        var g0 = perf_counter_ns()
        for layer in range(PK_LAYERS):
            stt.model.layers[layer].ff1_a.forward["cpu", T](TensorRefs[1](stt.enc.h), stt.enc.ff)
            stt.model.layers[layer].ff1_b.forward["cpu", T](TensorRefs[1](stt.enc.ff), stt.enc.y)
            stt.model.layers[layer].q.forward["cpu", T](TensorRefs[1](stt.enc.h), stt.enc.q)
            stt.model.layers[layer].k.forward["cpu", T](TensorRefs[1](stt.enc.h), stt.enc.k)
            stt.model.layers[layer].v.forward["cpu", T](TensorRefs[1](stt.enc.h), stt.enc.v)
            stt.model.layers[layer].rel_k.forward["cpu", P](TensorRefs[1](stt.enc.pos), stt.enc.relk)
            stt.model.layers[layer].o.forward["cpu", T](TensorRefs[1](stt.enc.att), stt.enc.y)
            stt.model.layers[layer].pw1.forward["cpu", T](TensorRefs[1](stt.enc.h), stt.enc.glu)
            stt.model.layers[layer].pw2.forward["cpu", T](TensorRefs[1](stt.enc.conv), stt.enc.y)
            stt.model.layers[layer].ff2_a.forward["cpu", T](TensorRefs[1](stt.enc.h), stt.enc.ff)
            stt.model.layers[layer].ff2_b.forward["cpu", T](TensorRefs[1](stt.enc.ff), stt.enc.y)
        var g1 = perf_counter_ns()
        print(files[f], " ", stt.tokenizer.decode(trace.text_ids()))
        print("  ", n_mel, "mel frames,", valid, "of", T, "encoder frames,",
              len(trace.tokens) - 1, "decode steps")
        print("  front end", ms(p0, p1), "ms | subsampling", ms(p1, p2),
              "ms | 24 blocks", ms(p2, p3), "ms (their GEMMs alone:", ms(g0, g1),
              "ms) | projection + decode", ms(p3, p4), "ms | total", ms(p0, p4), "ms")
