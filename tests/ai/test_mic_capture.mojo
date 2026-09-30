# +--------------------------------------------------------------------------+ #
# | Continuous capture: never blocks, keeps real time, stops cleanly
# +--------------------------------------------------------------------------+ #
"""Gate `MicCapture` / `rms` in `noeira/ai/audio_io.mojo`.

    pixi run mojo run -I . tests/ai/test_mic_capture.mojo

Hermetic: no microphone and no permission prompt. The capture device is
replaced by ffmpeg's own generators paced at real time (`-re -f lavfi`), so
the same pipe, pid and non-blocking read path runs as with a mic.

⚠ THE POINTS ARE THE ONES A VAD LOOP DEPENDS ON:

* **`read` never blocks** — in steady state a call costs well under half a
  millisecond while the loop runs at ~100 Hz; the worst call is the startup
  burst being converted, not a wait.
* **The stream keeps real time** — ~16 000 samples per second arrive in
  steady state, none lost or duplicated across reads that end mid-sample.
* **The level is right** — ffmpeg's `sine` is 1/8 full scale (RMS 0.0884),
  `anullsrc` is silence: the numbers a threshold is set against.
* **`stop` is fast and leaves no process**, and so does dropping a capture
  without calling it — the destructor must not hang in `pclose`.
* **A dead ffmpeg raises** instead of reading as silence forever.
"""

from std.ffi import external_call
from std.time import perf_counter_ns, sleep

from noeira.ai.audio_io import MicCapture, Pcm16Decoder, rms


comptime SINE = "-re -readrate_initial_burst 0 -f lavfi -i sine=frequency=440:sample_rate=16000"
comptime SILENCE = "-re -readrate_initial_burst 0 -f lavfi -i anullsrc=r=16000:cl=mono"


def _check(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAIL: " + what)


def _alive(pid: Int) -> Bool:
    return external_call["kill", Int32](Int32(pid), Int32(0)) == 0


def _ms(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e6


def _odd_splits() raises:
    """Every sample must survive chunks that end mid-sample."""
    var want: List[Int16] = [0, 1, -1, 32767, -32768, 12345, -12345, 256, -256, 7]
    var bytes = List[UInt8]()
    for i in range(len(want)):
        var v = Int(want[i]) & 0xFFFF
        bytes.append(UInt8(v & 0xFF))
        bytes.append(UInt8(v >> 8))
    # Chunk sizes 1, 2, 3, 1, 5, ... — odd boundaries everywhere.
    var sizes: List[Int] = [1, 2, 3, 1, 5, 1, 1, 4, 2]
    var dec = Pcm16Decoder()
    var got = List[Int16]()
    var pos = 0
    for k in range(len(sizes)):
        var chunk = List[UInt8]()
        for j in range(sizes[k]):
            if pos + j < len(bytes):
                chunk.append(bytes[pos + j])
        dec.feed(chunk, len(chunk), got)
        pos += sizes[k]
    _check(got == want, "odd-split decode lost or garbled samples")
    _check(dec.carry == -1, "a byte left over at the end")
    print("  s16le decode across odd chunk boundaries (incl. +/-32768/32767)   ok")


def main() raises:
    print("=== mic capture ===")
    _odd_splits()

    # ── 1. a tone, read from a 100 Hz loop for ~1.2 s ───────────────────
    var mic = MicCapture.start(16000, input_args=String(SINE))
    var pid = mic.pid
    _check(_alive(pid), "ffmpeg not running after start")
    var all = List[Int16]()
    var worst = 0.0
    var worst_steady = 0.0
    var t_first = -1.0
    var n_at_400 = -1
    var t0 = perf_counter_ns()
    while _ms(t0) < 1400.0:
        var tr = perf_counter_ns()
        var chunk = mic.read()
        var ms = _ms(tr)
        if ms > worst:
            worst = ms
        if _ms(t0) > 400.0 and ms > worst_steady:
            worst_steady = ms
        if len(chunk) > 0 and t_first < 0.0:
            t_first = _ms(t0)
        for i in range(len(chunk)):
            all.append(chunk[i])
        if n_at_400 < 0 and _ms(t0) >= 400.0:
            n_at_400 = len(all)
        sleep(0.01)
    # STEADY STATE ONLY: lavfi front-loads ~0.6 s at startup even with
    # `-readrate_initial_burst 0` (measured: 10 240 samples in the first
    # 105 ms, then 16 055/s). A microphone has no such burst.
    var got = len(all)
    var rate = Float64(got - n_at_400) / ((_ms(t0) - 400.0) / 1e3)
    # The cost of a read is converting what is waiting, so the startup burst
    # (10k samples) is the worst call; at 100 Hz a steady read holds ~160.
    _check(worst < 5.0, "read() blocked: worst call " + String(worst) + " ms")
    _check(worst_steady < 0.5, "steady-state read took " + String(worst_steady) + " ms")
    _check(rate > 13000.0 and rate < 19000.0, "not real time: " + String(rate) + " samples/s")
    var level = rms(all)
    _check(level > 0.080 and level < 0.097, "sine RMS " + String(level) + " (expect 0.0884)")
    # Continuity: a 440 Hz sine sampled at 16 kHz changes by at most
    # 2*pi*440/16000 * 4096 = 708 per sample. A dropped or doubled byte (a
    # read ending mid-sample, mishandled) breaks that by thousands.
    var jumps = 0
    for i in range(1, len(all)):
        var d = Int(all[i]) - Int(all[i - 1])
        if d > 800 or d < -800:
            jumps += 1
    _check(jumps == 0, String(jumps) + " discontinuities: samples lost or misaligned")
    var ts = perf_counter_ns()
    mic.stop()
    var stop_ms = _ms(ts)
    sleep(0.05)
    _check(stop_ms < 500.0, "stop() took " + String(stop_ms) + " ms")
    _check(not _alive(pid), "ffmpeg still running after stop()")
    print(
        "  tone: " + String(got) + " samples, " + String(Int(rate)) + "/s, rms "
        + String(level) + ", first audio after " + String(Int(t_first))
        + " ms, worst read " + String(worst) + " ms (steady " + String(worst_steady) + "), stop " + String(Int(stop_ms)) + " ms   ok"
    )

    # ── 2. silence reads as silence ─────────────────────────────────────
    var quiet = MicCapture.start(16000, input_args=String(SILENCE))
    var qs = List[Int16]()
    var tq = perf_counter_ns()
    while _ms(tq) < 500.0:
        var c = quiet.read()
        for i in range(len(c)):
            qs.append(c[i])
        sleep(0.01)
    quiet.stop()
    _check(len(qs) > 4000 and rms(qs) < 1e-4, "silence: " + String(len(qs)) + " samples, rms " + String(rms(qs)))
    print("  silence: " + String(len(qs)) + " samples, rms " + String(rms(qs)) + "   ok")

    # ── 3. dropped without stop(): the destructor must not hang ─────────
    var dropped_pid = 0
    var td = perf_counter_ns()
    # Mojo destroys a value at its LAST USE, so `tmp` dies right after read().
    var tmp = MicCapture.start(16000, input_args=String(SINE))
    dropped_pid = tmp.pid
    sleep(0.2)
    _ = tmp.read()
    var drop_ms = _ms(td)
    sleep(0.05)
    _check(drop_ms < 1500.0, "dropping a capture took " + String(drop_ms) + " ms")
    _check(not _alive(dropped_pid), "a dropped capture left ffmpeg running")
    print("  dropped without stop(): reaped in " + String(Int(drop_ms)) + " ms   ok")

    # ── 4. an ffmpeg that dies raises, rather than reading as silence ───
    var bad = MicCapture.start(16000, input_args=String("-f lavfi -i no_such_filter"))
    var raised = String("")
    var tb = perf_counter_ns()
    while _ms(tb) < 3000.0 and raised.byte_length() == 0:
        try:
            _ = bad.read()
        except e:
            raised = String(e)
        sleep(0.01)
    _check("stopped producing audio" in raised, "a dead ffmpeg did not raise: '" + raised + "'")
    print("  dead ffmpeg: raised after " + String(Int(_ms(tb))) + " ms   ok")
    print("=== mic capture: 14 checks passed ===")
