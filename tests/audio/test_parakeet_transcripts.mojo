# +--------------------------------------------------------------------------+ #
# | Parakeet in Mojo, end to end: WAV in, the reference's transcript out
# +--------------------------------------------------------------------------+ #
"""The task metric of the port (docs/NN_MODEL_PORTS_PLAN.md §6 step 4).

    pixi run -e detect python tools/parakeet/transcribe_reference.py \
        --manifest ~/.cache/noeira/parakeet/clips/manifest.tsv \
        --out ~/.cache/noeira/parakeet/clips/ref_parakeet_v3.tsv
    pixi run -e detect python tools/parakeet/transcribe_reference.py --pad-s 0.5 \
        --manifest ~/.cache/noeira/parakeet/clips/manifest.tsv \
        --out ~/.cache/noeira/parakeet/clips/ref_parakeet_v3_pad05.tsv
    pixi run mojo run -I . tests/audio/test_parakeet_transcripts.mojo [clips_dir]

Every clip goes WAV -> `ParakeetSTT.transcribe` (our front end, encoder,
decoder, tokenizer; no reference array anywhere) in two forms — as cut, and
with 0.5 s of silence on both sides — and each transcript must EQUAL the one
transformers' `ParakeetForTDT` gave for the same form, character for character.
That includes the reference's mistakes: this gate is about being the same
model, not about being right.

Non-vacuity: the two forms are known to differ on several clips, so the "as
cut" transcripts are also compared with the PADDED reference and that
comparison must FAIL on at least one clip. If it passed, the comparison would
not be telling the two apart.
"""

from std.os import getenv
from std.os.path import exists
from std.sys import argv
from std.time import perf_counter_ns

from noeira.io.wav import load_wav
from noeira.audio.parakeet.stt import ParakeetSTT


def read_text(path: String) raises -> String:
    var f = open(path, "r")
    var s = f.read()
    f.close()
    return s^


def reference_column(path: String) raises -> List[String]:
    """The transcripts of a `transcribe_reference.py --out` file, in order:
    the field after `=>`. `file`s are returned interleaved: [file, hyp, ...]."""
    var out = List[String]()
    # ⚠ `split("\n")`, not `splitlines()`: the fields are TAB-separated and
    # `splitlines()` broke these lines at the first tab.
    for line in read_text(path).split("\n"):
        var s = String(line)
        if s == "":
            continue
        var parts = s.split("\t")
        if len(parts) < 6 or String(parts[5]) != "=>":
            raise Error("malformed reference line: " + s)
        out.append(String(parts[0]))
        out.append(String(parts[6]) if len(parts) > 6 else String(""))
    return out^


def main() raises:
    var args = argv()
    var dir = getenv("HOME") + "/.cache/noeira/parakeet/clips"
    if len(args) >= 2:
        dir = String(args[1])
    var plain_path = dir + "/ref_parakeet_v3.tsv"
    var padded_path = dir + "/ref_parakeet_v3_pad05.tsv"
    if not exists(plain_path) or not exists(padded_path):
        raise Error(
            "no reference transcripts in " + dir
            + " — generate them with tools/parakeet/transcribe_reference.py"
        )
    var plain = reference_column(plain_path)
    var padded = reference_column(padded_path)
    if len(plain) != len(padded) or len(plain) == 0:
        raise Error("the two reference files do not list the same clips")

    print("Parakeet in Mojo vs transformers, transcripts (" + String(len(plain) // 2) + " clips, two forms)")
    var stt = ParakeetSTT[801].load()
    var fails = 0
    var crossed = 0
    var total_ns = 0
    var total_audio = 0.0
    for i in range(len(plain) // 2):
        var name = plain[2 * i]
        var wav = load_wav(dir + "/" + name)
        if wav.sample_rate != 16000 or wav.channels != 1:
            raise Error(name + ": need 16 kHz mono")
        var audio = wav.to_float32()

        var t0 = perf_counter_ns()
        var r0 = stt.transcribe(audio)
        var t1 = perf_counter_ns()
        var r1 = stt.transcribe(audio, 0.5)
        var t2 = perf_counter_ns()
        total_ns += Int(t2 - t0)
        total_audio += 2.0 * wav.duration_s() + 1.0

        var ok0 = r0.text == plain[2 * i + 1]
        var ok1 = r1.text == padded[2 * i + 1]
        if r0.text != padded[2 * i + 1]:
            crossed += 1
        print(
            "  " + ("PASS" if ok0 and ok1 else "FAIL") + "  " + name + "  "
            + String(Int(t1 - t0) // 1000000) + " + " + String(Int(t2 - t1) // 1000000)
            + " ms  '" + r0.text + "' | padded '" + r1.text + "'"
        )
        if not ok0:
            fails += 1
            print("        as cut, reference: '" + plain[2 * i + 1] + "'")
        if not ok1:
            fails += 1
            print("        padded, reference: '" + padded[2 * i + 1] + "'")

    var caught = crossed > 0
    print(
        "  " + ("PASS" if caught else "FAIL") + "  non-vacuity: " + String(crossed)
        + " 'as cut' transcripts differ from the PADDED reference"
    )
    if not caught:
        fails += 1
    print(
        "  " + String(total_ns // 1000000) + " ms for " + String(total_audio)
        + " s of audio (CPU fp32, compiled for " + String(ParakeetSTT[801].max_seconds())
        + " s)"
    )
    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("parakeet transcript gate failed")
