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

fp32, compiled for clips of up to 8 s. The encoder runs on the CPU, or on the
GPU when built for it — the target is a BUILD-time choice (one instantiation of
the model is two minutes of compile):

    pixi run -e apple mojo build -D PK_GPU=1 -I . examples/audio/parakeet_transcribe.mojo \
        -o build/parakeet_transcribe_gpu                  (-e nvidia on a CUDA machine)
"""

from std.sys import argv, default_accelerator
from std.sys.defines import is_defined, get_defined_int
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.io.wav import load_wav
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.audio.parakeet.config import PK_D, PK_FF, PK_LAYERS
from noeira.audio.parakeet.decode import pk_greedy_decode
from noeira.audio.parakeet.encoder import (
    PkEncoderWork, pk_subsample, pk_block, _silu_any, _gemm_any, _ln_any,
)
from noeira.audio.parakeet.stt import ParakeetSTT


comptime F_MAX: Int = get_defined_int["PK_F_MAX", 801]()
"""Mel frames compiled for: 801 (8 s) unless built with `-D PK_F_MAX=401` (4 s)."""
comptime TARGET: StaticString = "gpu" if is_defined["PK_GPU"]() else "cpu"
comptime WEIGHTS: StaticString = "fp16" if is_defined["PK_FP16"]() else "fp32"
"""`-D PK_FP16=1` (NVIDIA, with `PK_GPU`): half-precision GEMM weights."""
comptime F_SHORT: Int = 401 if (is_defined["PK_GPU"]() and default_accelerator().is_nvidia_gpu() and F_MAX > 401) else 0
"""On NVIDIA: a second, 4 s shape for the clips that fit (`stt.mojo`). The
`--profile` breakdown below always times the 8 s shape."""


def sync(ctx: Optional[DeviceContext]) raises:
    """Wait for the device, so a stage's time is its own."""
    if ctx:
        ctx.value().synchronize()


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
    var stt = ParakeetSTT[F_MAX, TARGET, F_SHORT, WEIGHTS].load()
    print("loaded in", Int(ms(t0, perf_counter_ns())), "ms; encoder on the", TARGET,
          "; compiled for", ParakeetSTT[F_MAX, TARGET, F_SHORT, WEIGHTS].max_seconds(), "s")

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
        var valid = pk_subsample[F_MAX, TARGET](stt.model, stt.enc, n_mel, stt.ctx)
        sync(stt.ctx)
        var p2 = perf_counter_ns()
        for layer in range(PK_LAYERS):
            pk_block[F_MAX, TARGET, WEIGHTS](stt.model, stt.enc, layer, valid, stt.ctx)
        sync(stt.ctx)
        var p3 = perf_counter_ns()
        comptime T = PkEncoderWork[F_MAX].T
        stt.model.enc_proj.forward[TARGET, T](
            TensorRefs[1](stt.enc.x), stt.enc.proj, stt.ctx
        )
        sync(stt.ctx)
        var p4 = perf_counter_ns()
        var trace = pk_greedy_decode[TARGET, T](
            stt.model, stt.enc.proj, valid, T, stt.dec, stt.ctx
        )
        var p5 = perf_counter_ns()
        # The GEMMs of the 24 blocks alone, on whatever the buffers hold.
        var g0 = perf_counter_ns()
        for layer in range(PK_LAYERS if WEIGHTS == "fp32" else 0):
            _gemm_any[TARGET, T, PK_D, PK_FF](stt.model.layers[layer].ff1_a, stt.enc.h, stt.enc.ff, valid, stt.ctx)
            _gemm_any[TARGET, T, PK_FF, PK_D](stt.model.layers[layer].ff1_b, stt.enc.ff, stt.enc.y, valid, stt.ctx)
            _gemm_any[TARGET, T, PK_D, PK_D](stt.model.layers[layer].q, stt.enc.h, stt.enc.q, valid, stt.ctx)
            _gemm_any[TARGET, T, PK_D, PK_D](stt.model.layers[layer].k, stt.enc.h, stt.enc.k, valid, stt.ctx)
            _gemm_any[TARGET, T, PK_D, PK_D](stt.model.layers[layer].v, stt.enc.h, stt.enc.v, valid, stt.ctx)
            _gemm_any[TARGET, T, PK_D, PK_D](stt.model.layers[layer].o, stt.enc.att, stt.enc.y, valid, stt.ctx)
            _gemm_any[TARGET, T, PK_D, 2 * PK_D](stt.model.layers[layer].pw1, stt.enc.h, stt.enc.glu, valid, stt.ctx)
            _gemm_any[TARGET, T, PK_D, PK_D](stt.model.layers[layer].pw2, stt.enc.conv, stt.enc.y, valid, stt.ctx)
            _gemm_any[TARGET, T, PK_D, PK_FF](stt.model.layers[layer].ff2_a, stt.enc.h, stt.enc.ff, valid, stt.ctx)
            _gemm_any[TARGET, T, PK_FF, PK_D](stt.model.layers[layer].ff2_b, stt.enc.ff, stt.enc.y, valid, stt.ctx)
        sync(stt.ctx)
        var g1 = perf_counter_ns()
        # The five LayerNorms of each block alone.
        for layer in range(PK_LAYERS):
            _ln_any[TARGET, T](stt.model.layers[layer].norm_ff1, stt.enc.x, stt.enc.h, stt.enc.ln_xhat, stt.enc.ln_inv, valid, stt.ctx)
            _ln_any[TARGET, T](stt.model.layers[layer].norm_att, stt.enc.x, stt.enc.h, stt.enc.ln_xhat, stt.enc.ln_inv, valid, stt.ctx)
            _ln_any[TARGET, T](stt.model.layers[layer].norm_conv, stt.enc.x, stt.enc.h, stt.enc.ln_xhat, stt.enc.ln_inv, valid, stt.ctx)
            _ln_any[TARGET, T](stt.model.layers[layer].norm_ff2, stt.enc.x, stt.enc.h, stt.enc.ln_xhat, stt.enc.ln_inv, valid, stt.ctx)
            _ln_any[TARGET, T](stt.model.layers[layer].norm_out, stt.enc.x, stt.enc.h, stt.enc.ln_xhat, stt.enc.ln_inv, valid, stt.ctx)
        sync(stt.ctx)
        var g2 = perf_counter_ns()
        # 1000 launches of the smallest kernel: the cost of a launch here.
        for _ in range(1000):
            _silu_any[TARGET, T * 1024](stt.enc.y, T * 1024, stt.ctx)
        sync(stt.ctx)
        var g3 = perf_counter_ns()
        print(files[f], " ", stt.tokenizer.decode(trace.text_ids()))
        print("  ", n_mel, "mel frames,", valid, "of", T, "encoder frames,",
              len(trace.tokens) - 1, "decode steps")
        print("  front end", ms(p0, p1), "ms | subsampling", ms(p1, p2),
              "ms | 24 blocks", ms(p2, p3), "ms | projection", ms(p3, p4),
              "ms | decode", ms(p4, p5), "ms | total", ms(p0, p5), "ms")
        print("  alone: the blocks' 240 GEMMs (fp32 weights only)", ms(g0, g1), "ms | their 120 LayerNorms",
              ms(g1, g2), "ms | 1000 x one elementwise pass over [T, 1024]", ms(g2, g3), "ms")
