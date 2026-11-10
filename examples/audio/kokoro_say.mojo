# +--------------------------------------------------------------------------+ #
# | Text-to-speech on this machine: Kokoro-82M in Mojo
# +--------------------------------------------------------------------------+ #
"""Speak text with `hexgrad/Kokoro-82M`, no Python, no server.

    pixi run mojo build -I . examples/audio/kokoro_say.mojo -o build/kokoro_say
    ./build/kokoro_say --voice ff_siwis -o bonjour.wav "Bonjour, comment puis-je vous aider ?"
    ./build/kokoro_say --voice af_heart -o fridge.wav "I am going to the fridge."
    ./build/kokoro_say --phonemes --voice af_heart "aɪɐm ɡˌoʊɪŋ tə ðə fɹˈɪdʒ."

Needs, once per machine, the converted checkpoint and espeak-ng:

    pixi run -e detect python tools/kokoro/convert_kokoro_checkpoint.py
    python3 tools/kokoro/fetch_espeak_ng.py

The voice's first letter picks the language (`a` American English, `b`
British, `f` French, ...). `--g2p deployment|misaki|misaki_en` overrides the
text-to-phoneme mapping, which defaults to kokoro-onnx's for English and
upstream Kokoro's for the rest (`noeira/audio/kokoro/g2p.mojo` says why).
Output: 16-bit mono WAV at 24 kHz.

CPU by default (fp32; it needs Accelerate's GEMM, so macOS). On a GPU — the
Orin's, or any NVIDIA or Metal device — build with `-D KOKORO_GPU=1` (and
`-D KOKORO_TF32=1` for the vocoder's GEMMs in TF32, NVIDIA):

    pixi run -e jetson build-jetson -D KOKORO_GPU=1 examples/audio/kokoro_say.mojo -o build/kokoro_say_gpu
"""

from std.sys import argv
from std.sys.defines import is_defined
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.io.wav import wav_from_float32, save_wav
from noeira.audio.espeak import EspeakNG
from noeira.audio.kokoro.g2p import KK_G2P_DEPLOYMENT, KK_G2P_MISAKI, KK_G2P_MISAKI_EN
from noeira.audio.kokoro.device import KokoroDevice
from noeira.audio.kokoro.tts import KokoroTTS, KokoroSpeech


comptime GPU: Bool = is_defined["KOKORO_GPU"]()


def main() raises:
    var args = argv()
    var voice = String("af_heart")
    var out = String("kokoro.wav")
    var speed = Float32(1.0)
    var seed = UInt64(0)
    var mode = -1
    var as_phonemes = False
    var text = String("")
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
        elif a == "--phonemes":
            as_phonemes = True
        elif a == "--g2p":
            i += 1
            var g = String(args[i])
            if g == "deployment":
                mode = KK_G2P_DEPLOYMENT
            elif g == "misaki":
                mode = KK_G2P_MISAKI
            elif g == "misaki_en":
                mode = KK_G2P_MISAKI_EN
            else:
                raise Error("--g2p is deployment, misaki or misaki_en")
        else:
            text = a^
        i += 1
    if text == "":
        print("usage: kokoro_say [--voice af_heart] [--speed 1.0] [--seed 0] [--g2p deployment]"
              " [--phonemes] [-o out.wav] \"<text>\"")
        return

    var t0 = perf_counter_ns()
    var s: KokoroSpeech
    var t1: Int
    var t2: Int
    var t3: Int
    comptime if GPU:
        var k = KokoroDevice.load(DeviceContext())
        t1 = perf_counter_ns()
        var es = EspeakNG.open()
        # A first, untimed pass: the device's first launches pay their setup.
        _ = k.speak(es, String("Hello."), String("af_heart"))
        t2 = perf_counter_ns()
        if as_phonemes:
            s = k.synthesize(text, voice, speed, seed)
        else:
            s = k.speak(es, text, voice, speed, seed, mode)
        t3 = perf_counter_ns()
    else:
        var tts = KokoroTTS.load()
        t1 = perf_counter_ns()
        if as_phonemes:
            t2 = perf_counter_ns()
            s = tts.synthesize(text, voice, speed, seed)
        else:
            var es = EspeakNG.open()
            t2 = perf_counter_ns()
            s = tts.speak(es, text, voice, speed, seed, mode)
        t3 = perf_counter_ns()
    save_wav(out, wav_from_float32(s.audio, 24000))
    print(out, ":", Float64(len(s.audio)) / 24000.0, "s,", s.frames, "frames, voice", voice)
    print("  phonemes:", s.phonemes)
    print("  load", (t1 - t0) // 1000000, "ms, text to speech", (t3 - t2) // 1000000, "ms",
          "(GPU)" if GPU else "(CPU)")
