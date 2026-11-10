# +--------------------------------------------------------------------------+ #
# | Text-to-speech on this machine: Kokoro-82M in Mojo
# +--------------------------------------------------------------------------+ #
"""Speak a PHONEME string with `hexgrad/Kokoro-82M`, no Python, no server.

    pixi run mojo build -I . examples/audio/kokoro_say.mojo -o build/kokoro_say
    ./build/kokoro_say --voice ff_siwis -o bonjour.wav "bɔ̃ʒˈuʁ, kɔmˌɑ̃ pyˈiʒ vuz ɛdˈe ?"
    ./build/kokoro_say --voice af_heart -o fridge.wav "aɪɐm ɡˌoʊɪŋ tə ðə fɹˈɪdʒ."

Needs the converted checkpoint (`tools/kokoro/convert_kokoro_checkpoint.py`,
once, in the `detect` environment). The input is phonemes in Kokoro's alphabet,
as espeak-ng writes them; text input comes with the espeak-ng binding
(docs/NN_MODEL_PORTS_PLAN.md §11.3). Output: 16-bit mono WAV at 24 kHz.

CPU, fp32: the reference path, about real time on an M1.
"""

from std.sys import argv
from std.time import perf_counter_ns

from noeira.io.wav import wav_from_float32, save_wav
from noeira.audio.kokoro.tts import KokoroTTS


def main() raises:
    var args = argv()
    var voice = String("af_heart")
    var out = String("kokoro.wav")
    var speed = Float32(1.0)
    var seed = UInt64(0)
    var phonemes = String("")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--voice":
            i += 1
            voice = String(args[i])
        elif a == "-o":
            i += 1
            out = String(args[i])
        elif a == "--speed":
            i += 1
            speed = Float32(atof(args[i]))
        elif a == "--seed":
            i += 1
            seed = UInt64(atol(args[i]))
        else:
            phonemes = a^
        i += 1
    if phonemes == "":
        print("usage: kokoro_say [--voice af_heart] [--speed 1.0] [--seed 0] [-o out.wav] \"<phonemes>\"")
        return

    var t0 = perf_counter_ns()
    var tts = KokoroTTS.load()
    var t1 = perf_counter_ns()
    var s = tts.synthesize(phonemes, voice, speed, seed)
    var t2 = perf_counter_ns()
    var wav = wav_from_float32(s.audio, 24000)
    save_wav(out, wav)
    print(
        out, ":", Float64(len(s.audio)) / 24000.0, "s of speech,", s.frames, "frames, voice", voice,
        "| load", Int(t1 - t0) // 1000000, "ms, synthesis", Int(t2 - t1) // 1000000, "ms",
    )
