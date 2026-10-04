# +--------------------------------------------------------------------------+ #
# | The OS voice: any text, never blocks, an exact end signal, no overlap
# +--------------------------------------------------------------------------+ #
"""Gate `LocalVoice` / `speak_local`'s command in `noeira/ai/audio_io.mojo`.

    pixi run mojo run -I . tests/ai/test_local_voice.mojo      # macOS (`say`)

SILENT: every utterance renders to a file (`say -o`), so nothing plays on the
speakers and the process lifecycle is the same as speaking aloud.

⚠ THE POINTS ARE THE ONES A VOICE DEMO BROKE ON:

* **Text with apostrophes, quotes and accents is spoken, not refused.**
  `quote_arg` refuses a single quote, and French is full of them — "je
  m'accroupis" took the G1 demo down. The text now goes through a file.
* **`say` returns at once and `speaking()` never blocks.**
* **`speaking()` turns False when the audio has ENDED, not on a timer** — the
  rendered file is complete when it does.
* **A new utterance cuts off the previous one** (no two voices sharing one
  "done"), and `stop()` leaves no process behind.
"""

from std.ffi import external_call
from std.os.path import exists
from std.sys import CompilationTarget
from std.time import perf_counter_ns, sleep

from noeira.ai.audio_io import LocalVoice
from noeira.io.fileio import file_size, remove_file


def _check(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAIL: " + what)


def _alive(pid: Int) -> Bool:
    return external_call["kill", Int32](Int32(pid), Int32(0)) == 0


def _ms(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e6


def main() raises:
    comptime if not CompilationTarget.is_macos():
        print("=== local voice: skipped (the gate renders with macOS `say -o`) ===")
        return
    print("=== local voice ===")
    comptime OUT = "/tmp/noeira_voice_gate.aiff"
    try:
        remove_file(String(OUT))
    except:
        pass

    # ── 1. hostile text: apostrophes, both quotes, accents, a backslash, $ ──
    var voice = LocalVoice()
    var t0 = perf_counter_ns()
    voice.say(
        String("J'ai fini, c'est prêt : « d'accord » \"oui\" \\ $HOME `ls`"),
        String(OUT),
    )
    var start_ms = _ms(t0)
    var pid = voice._pid
    _check(voice.speaking(), "not speaking right after say()")
    _check(start_ms < 300.0, "say() blocked for " + String(start_ms) + " ms")
    var worst = 0.0
    while True:
        var tp = perf_counter_ns()
        var s = voice.speaking()
        var ms = _ms(tp)
        if ms > worst:
            worst = ms
        if not s:
            break
        _check(_ms(t0) < 15000.0, "never finished")
        sleep(0.005)
    var done_ms = _ms(t0)
    # A call that WAITED would last the utterance (~seconds). The threshold
    # separates that from scheduler noise: one call measured 22 ms while
    # another session compiled on the same machine, typical ones 0.2 ms.
    _check(worst < 100.0, "speaking() blocked for " + String(worst) + " ms")
    _check(not _alive(pid), "`say` still running after speaking() went False")
    _check(exists(OUT) and file_size(String(OUT)) > 10000, "no audio rendered for the hostile text")
    print(
        "  hostile text (' \" « » \\ $ `): rendered, say() " + String(Int(start_ms))
        + " ms, done at " + String(Int(done_ms)) + " ms, worst speaking() "
        + String(worst) + " ms   ok"
    )

    # ── 2. a new utterance cuts the previous one off ────────────────────
    var long_text = String("")
    for _ in range(40):
        long_text += "Le robot marche en avant, puis il tourne à gauche. "
    voice.say(long_text, String(OUT))
    var first = voice._pid
    sleep(0.1)
    voice.say(String("Stop."), String(OUT))
    sleep(0.05)
    _check(not _alive(first), "the first utterance kept running under the second")
    voice.wait()
    print("  overlap: the second say() killed the first   ok")

    # ── 3. stop() mid-utterance ─────────────────────────────────────────
    voice.say(long_text, String(OUT))
    var p3 = voice._pid
    sleep(0.1)
    var ts = perf_counter_ns()
    voice.stop()
    var stop_ms = _ms(ts)
    sleep(0.05)
    _check(not voice.speaking(), "speaking() after stop()")
    _check(not _alive(p3), "`say` survived stop()")
    _check(stop_ms < 200.0, "stop() took " + String(stop_ms) + " ms")
    print("  stop(): " + String(stop_ms) + " ms, no process left   ok")
    try:
        remove_file(String(OUT))
    except:
        pass
    print("=== local voice: 10 checks passed ===")
