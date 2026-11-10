"""The speech services in one process: `ParakeetService` against transformers,
then Parakeet and Kokoro side by side under a 50 Hz caller.

    pixi run -e apple mojo build -I . tests/audio/test_speech_services_gpu.mojo -o build/x && ./build/x
    (the Orin: `build-jetson ... -D KOKORO_TF32=1`, JetPack's cuBLAS and cuDNN first)

Needs Parakeet's checkpoint and clips with their reference transcripts
(`tests/manifests/parakeet-ref.txt`), Kokoro's checkpoint and espeak-ng
(`tests/manifests/kokoro-ref.txt`).

A. `ParakeetService` (worker thread, 0.5 s padding by default) gives
   transformers' padded transcript on every clip (`ref_parakeet_v3_pad05.tsv`).
   Non-vacuity: the same clips with the padding off must differ from it on
   some clips — so the service's padding is really applied.
B. Both models loaded in ONE process, each on its own thread and device
   context, and a caller ticking at 50 Hz: Kokoro speaks a sentence, its WAV
   goes to Parakeet, and the transcript must be the sentence. Every `start`
   and `poll` the caller makes is timed — the worst must stay under 2 ms.
C. How much memory the two services take (Linux: `MemAvailable`).
"""

from std.os import getenv
from std.sys import default_accelerator
from std.os.path import exists
from std.time import perf_counter_ns

from noeira.ai.local_service import SVC_LOADING, SVC_READY
from noeira.audio.kokoro.service import KokoroService
from noeira.audio.parakeet.service import ParakeetService
from noeira.core.concurrent.thread import sleep_us
from noeira.io.wav import decode_wav, load_wav


comptime IS_NV = default_accelerator().is_nvidia_gpu()
comptime PK_SHORT: Int = 401 if IS_NV else 0
comptime PK_W: StaticString = "fp16" if IS_NV else "fp32"
comptime STT = ParakeetService[1101, "gpu", PK_SHORT, PK_W]
comptime FRAME_US = 20_000


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  ok:", what)
    else:
        print("  FAIL:", what)
        fails += 1


def read_text(path: String) raises -> String:
    var f = open(path, "r")
    var s = f.read()
    f.close()
    return s^


def reference(path: String) raises -> List[String]:
    """[file, transcript, file, transcript, ...] of a `transcribe_reference.py` file."""
    var out = List[String]()
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


def mem_available_mb() -> Int:
    """Linux `MemAvailable` in MB, -1 elsewhere."""
    if exists("/proc/meminfo"):
        try:
            for line in read_text("/proc/meminfo").split("\n"):
                var s = String(line)
                if s.startswith("MemAvailable:"):
                    var kb = 0
                    for c in s.as_bytes():
                        if c >= 48 and c <= 57:
                            kb = kb * 10 + Int(c - 48)
                    return kb // 1024
        except:
            pass
    return -1


def normal(s: String) -> String:
    """Lower case, letters and digits and single spaces only."""
    var out = String("")
    var space = False
    for c in s.lower().codepoint_slices():
        var ch = String(c)
        var b = ch.as_bytes()[0]
        if (b >= 97 and b <= 122) or (b >= 48 and b <= 57) or b >= 128:
            if space and out.byte_length() > 0:
                out += " "
            space = False
            out += ch
        else:
            space = True
    return out^


struct Tick(Movable):
    """The 50 Hz caller's clock: the worst time any one call took."""

    var worst_us: Int
    var frames: Int

    def __init__(out self):
        self.worst_us = 0
        self.frames = 0

    def note(mut self, t0: Int):
        self.worst_us = max(self.worst_us, (perf_counter_ns() - t0) // 1000)


def main() raises:
    print("speech services in-process (Parakeet ", PK_W, ", short shape ", PK_SHORT, "; Kokoro)")
    var fails = 0
    var mem0 = mem_available_mb()

    # ── both services start loading at once ──
    var t_load = perf_counter_ns()
    var stt = STT.from_spec(String("native"))
    var tts = KokoroService.from_spec(String("native"), String("af_heart"))
    while (stt.state() == SVC_LOADING or tts.state() == SVC_LOADING) and perf_counter_ns() - t_load < 300_000_000_000:
        _ = sleep_us(10_000)
    check(fails, stt.state() == SVC_READY and tts.state() == SVC_READY,
          String("both READY after ") + String((perf_counter_ns() - t_load) // 1_000_000) + " ms (Parakeet "
          + String(stt.load_ms()) + " ms, Kokoro " + String(tts.load_ms()) + " ms, loaded side by side)")
    var mem1 = mem_available_mb()
    if mem0 >= 0:
        print("  memory: MemAvailable", mem0, "->", mem1, "MB (", mem0 - mem1, "MB for the two services )")

    # ── A: Parakeet behind the service == transformers, padded ──
    var dir = getenv("HOME") + "/.cache/noeira/parakeet/clips"
    var padded = reference(dir + "/ref_parakeet_v3_pad05.tsv")
    var n_ok = 0
    var n = len(padded) // 2
    var worst_ms = 0.0
    for i in range(n):
        var wav = load_wav(dir + "/" + padded[2 * i])
        stt.start(wav)
        while not stt.poll(5):
            pass
        var tr = stt.result()
        worst_ms = max(worst_ms, tr.latency_ms)
        if tr.text == padded[2 * i + 1]:
            n_ok += 1
        else:
            print("    ", padded[2 * i], ": '", tr.text, "' vs reference '", padded[2 * i + 1], "'")
    check(fails, n_ok == n, String(n_ok) + " / " + String(n) + " transcripts equal transformers' (padded); slowest "
          + String(worst_ms) + " ms")
    stt.pad_s = 0.0
    var differ = 0
    for i in range(n):
        var wav = load_wav(dir + "/" + padded[2 * i])
        stt.start(wav)
        while not stt.poll(5):
            pass
        if stt.result().text != padded[2 * i + 1]:
            differ += 1
    check(fails, differ > 0, String("non-vacuity: without the padding ") + String(differ)
          + " clips differ from the padded reference")
    stt.pad_s = 0.5

    # ── B: a 50 Hz caller, Kokoro then Parakeet, both in flight in turn ──
    var sentences: List[String] = [
        "I am going to the fridge.",
        "Turn left, then walk to the door.",
        "Raise your right hand, please.",
    ]
    var tick = Tick()
    var t_all = perf_counter_ns()
    for s in sentences:
        var t0 = perf_counter_ns()
        tts.start(s)
        tick.note(t0)
        var heard = String("")
        var phase = 0
        while phase < 2:
            tick.frames += 1
            if phase == 0:
                t0 = perf_counter_ns()
                var done = tts.poll()
                tick.note(t0)
                if done:
                    var wb = tts.result_wav_bytes()
                    var wav = decode_wav(wb)
                    t0 = perf_counter_ns()
                    stt.start(wav)          # the 24 kHz -> 16 kHz resampling is the caller's
                    tick.note(t0)
                    phase = 1
            else:
                t0 = perf_counter_ns()
                var done = stt.poll()
                tick.note(t0)
                if done:
                    heard = stt.result().text
                    phase = 2
            _ = sleep_us(FRAME_US)
        check(fails, normal(heard) == normal(s), String("round trip '") + s + "' -> '" + heard
              + "' (Kokoro " + String(tts.latency_ms()) + " ms, Parakeet " + String(stt.call.latency_ms()) + " ms)")
    check(fails, tick.worst_us < 2000, String("the caller's worst call: ") + String(tick.worst_us) + " us over "
          + String(tick.frames) + " frames (" + String((perf_counter_ns() - t_all) // 1_000_000) + " ms)")

    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
