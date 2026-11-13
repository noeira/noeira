"""A recording through the detector, two ways: a diagnostic, not a gate.

    pixi run mojo run -I . tests/robots/g1_vad_replay.mojo clip.wav [busy seconds after a segment, default 2.5] [offline]

1. OFFLINE: the file's samples straight into `G1Vad`, 20 ms at a time, the
   caller "busy" (no segment may open) for the given time after each segment,
   as the voice loop is while it transcribes, decides and replies.
2. LIVE: the same file through `MicCapture` (`ffmpeg -re -i clip.wav`), polled
   about 60 times a second, the same busy rule on the wall clock. Printed per
   segment: when it was delivered against when it lies in the file.

When the two disagree, the capture's delivery is the difference; when both miss
an utterance the detector or the recording is.

What it settled (2026-11-13, docs/ROBOT_BRAIN_PLAN.md §15): on the recording
whose second utterance the room demo lost, both ways found all seven. So the
loss was the loop's (it was busy), not the detector's or the capture's. The
live positions are 3.2 s earlier than the offline ones: `-fflags nobuffer`
drops that much from the head of a file (`MicCapture`'s note).
"""

from std.sys import argv
from std.time import perf_counter_ns

from noeira.ai.audio_io import MicCapture
from noeira.core.concurrent.thread import sleep_us
from noeira.envs.robots.g1_vad import G1Vad, VL_FRAME
from noeira.io.wav import load_wav


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: g1_vad_replay clip.wav [busy seconds] [offline]")
    var path = String(args[1])
    var busy_s = Float64(String(args[2])) if len(args) > 2 else 2.5
    var wav = load_wav(path)
    var n = len(wav.samples)
    print(path, "|", round(Float64(n) / 16000.0, 1), "s | busy", busy_s, "s after each segment")

    print("offline:")
    var vad = G1Vad(0.6, 0.35, 10.0)
    var at = 0
    var busy_until = 0
    while at < n:
        var part = List[Int16]()
        for i in range(at, min(at + VL_FRAME, n)):
            part.append(wav.samples[i])
        at += VL_FRAME
        var ev = vad.feed(part, at > 19200 and at >= busy_until, at < busy_until)
        if ev.note != "":
            print("   ", round(Float64(at) / 16000.0, 2), "s:", ev.note)
        if ev.done:
            print("    segment", round(Float64(vad.consumed - len(vad.seg)) / 16000.0, 2), "to", round(Float64(vad.consumed) / 16000.0, 2),
                  "s, peak", round(vad.peak, 3), "floor", round(vad.floor, 5))
            busy_until = at + Int(busy_s * 16000.0)

    if len(args) > 3 and String(args[3]) == "offline":
        return
    print("live (ffmpeg -re):")
    var mic = MicCapture.start(16000, String(""), "-re -i " + path)
    var live = G1Vad(0.6, 0.35, 10.0)
    var t0 = perf_counter_ns()
    var busy_until_ns = 0
    var biggest = 0
    var reads = 0
    while True:
        var pcm: List[Int16]
        try:
            pcm = mic.read()
        except e:
            break
        reads += 1
        if len(pcm) > biggest:
            biggest = len(pcm)
        var now = perf_counter_ns()
        var ev = live.feed(pcm, mic.seconds_read() > 1.2 and now >= busy_until_ns, now < busy_until_ns)
        if ev.note != "":
            print("   ", ev.note)
        if ev.done:
            print("    segment", round(Float64(live.consumed - len(live.seg)) / 16000.0, 2), "to", round(Float64(live.consumed) / 16000.0, 2),
                  "s of the file, delivered at", round(Float64(now - t0) / 1e9, 2), "s, peak", round(live.peak, 3))
            busy_until_ns = now + Int(busy_s * 1e9)
        _ = sleep_us(16_000)
    print("   ", reads, "reads, the largest", biggest, "samples;", round(mic.seconds_read(), 1), "s read in", round(Float64(perf_counter_ns() - t0) / 1e9, 1), "s")
