"""The detector on its own: the same samples give the same segments, however they arrive.

    pixi run mojo run -I . tests/robots/test_g1_vad.mojo

WHY THIS EXISTS. The detector used to run once a POLL inside `G1VoiceLoop`, on
whatever one microphone read returned, with its durations on the wall clock.
After a stall the audio comes back in a burst: in every headless run of the
room demo an utterance was lost or cut to its tail (`g1_vad.mojo`'s note).
`G1Vad` counts samples instead, and this gate holds it to the property that
was missing:

1. ONE RECORDING, FIVE DELIVERIES: 20 ms chunks (real time), odd-sized chunks,
   half-second chunks (a stalled loop), three-second chunks, and the whole
   recording in one call (the worst burst). The segments must be the SAME,
   sample for sample: where each starts, how long it is, its peak.
2. Each segment holds its utterance: it starts within the pre-roll before the
   speech and ends within the hang time after it.
3. What is not speech: a click of two frames opens nothing; a knock under the
   caller's minimum length, and a faint sound, open and are dropped with a
   note.
4. `may_open` false: nothing opens, the pre-roll stays bounded, the floor still
   tracks the room.
5. A caller that is busy after a segment (as the voice loop is while it
   transcribes) loses the speech that follows, and only that: the next
   utterance, heard when it is idle again, is whole.
6. The echo gate (`skip`): discarded audio leaves no pre-roll, and a segment in
   hand still closes after its quiet time.
7. The guards: a sound that never stops ends at `max_seg`; a forced end.
8. Speech while the caller says it is BUSY (`feed`'s third argument): lost
   whole and reported once, with its length, when it ends. A caller that comes
   back to idle in the MIDDLE of it gets no segment made of its tail (« Je
   m'appelle Denis. » was heard as « Denis. » that way), and the utterance
   after it is whole. The same audio WITHOUT the flag does send the tail: the
   fixture reaches the fault.

The recording is synthetic (a tone for speech, noise for the room) and
deterministic. Nothing here opens a device.
"""

from std.math import sin, pi
from std.sys import exit

from noeira.envs.robots.g1_vad import G1Vad, G1VadEvent, VL_FRAME, VL_PREROLL_S


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  ok  :", what)
    else:
        print("  FAIL:", what)
        fails += 1


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def unit(mut self) -> Float64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Float64((self.s >> 33) & 0xFFFFFF) / Float64(0x1000000) - 0.5


def room(mut out: List[Int16], seconds: Float64, mut rng: Rng):
    """Quiet-room noise, about 0.001 RMS."""
    for _ in range(Int(seconds * 16000.0)):
        out.append(Int16(Int(rng.unit() * 110.0)))


def speech(mut out: List[Int16], seconds: Float64, mut rng: Rng, amp: Float64 = 8000.0):
    """A 220 Hz tone with a little noise: about 0.17 RMS."""
    var n0 = len(out)
    for i in range(Int(seconds * 16000.0)):
        var v = amp * sin(2.0 * pi * 220.0 * Float64(n0 + i) / 16000.0) + rng.unit() * 110.0
        out.append(Int16(Int(v)))


@fieldwise_init
struct Seg(Copyable, Movable):
    var start: Int
    var length: Int
    var peak: Float64
    var note: String


def run(ref pcm: List[Int16], chunk: Int) -> List[Seg]:
    """Feed `pcm` in chunks of `chunk` samples to a detector whose caller is
    always idle, taking each finished segment at once."""
    var vad = G1Vad(0.6, 0.35, 10.0)
    var out = List[Seg]()
    var at = 0
    var n = len(pcm)
    while at < n:
        var part = List[Int16]()
        for i in range(at, min(at + chunk, n)):
            part.append(pcm[i])
        at += chunk
        var ev = vad.feed(part, True)
        while True:
            if ev.note != "":
                out.append(Seg(-1, 0, 0.0, ev.note.copy()))
            if not ev.done:
                break
            out.append(Seg(vad.consumed - len(vad.seg), len(vad.seg), vad.peak, String("")))
            var nothing = List[Int16]()
            ev = vad.feed(nothing, True)
    return out^


def play(mut vad: G1Vad, ref pcm: List[Int16], lo: Int, hi: Int, may_open: Bool, busy: Bool, mut out: List[Seg]):
    """Feed `pcm[lo:hi]` 20 ms at a time. Appended to `out`: each finished
    segment, and each note (start -1, the missed seconds in `peak`)."""
    var at = lo
    while at < hi:
        var part = List[Int16]()
        for i in range(at, min(at + VL_FRAME, hi)):
            part.append(pcm[i])
        at += VL_FRAME
        var ev = vad.feed(part, may_open, busy)
        if ev.note != "":
            out.append(Seg(-1, 0, ev.missed_s, ev.note.copy()))
        if ev.done:
            out.append(Seg(vad.consumed - len(vad.seg), len(vad.seg), vad.peak, String("")))


def same(a: List[Seg], b: List[Seg]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i].start != b[i].start or a[i].length != b[i].length or a[i].peak != b[i].peak or a[i].note != b[i].note:
            return False
    return True


def main() raises:
    var fails = 0
    var rng = Rng(20261113)
    var pcm = List[Int16]()
    room(pcm, 2.0, rng)
    var a0 = len(pcm)
    speech(pcm, 1.5, rng)          # utterance A
    var a1 = len(pcm)
    room(pcm, 3.0, rng)
    var b0 = len(pcm)
    speech(pcm, 2.0, rng)          # utterance B
    var b1 = len(pcm)
    room(pcm, 2.0, rng)
    speech(pcm, 0.04, rng)         # a click: two frames, under the three that open a segment
    room(pcm, 2.5, rng)
    var c0 = len(pcm)
    speech(pcm, 0.6, rng)          # utterance C, short
    var c1 = len(pcm)
    room(pcm, 2.0, rng)

    print("1. one recording (" + String(round(Float64(len(pcm)) / 16000.0, 1)) + " s), five deliveries")
    var real = run(pcm, VL_FRAME)
    var odd = run(pcm, 211)
    var stalled = run(pcm, 8000)
    var long_stall = run(pcm, 48000)
    var burst = run(pcm, len(pcm))
    for s in real:
        if s.start >= 0:
            print("      segment at", round(Float64(s.start) / 16000.0, 2), "s,", round(Float64(s.length) / 16000.0, 2), "s long, peak", round(s.peak, 3))
        else:
            print("     ", s.note)
    check(fails, same(real, odd), "odd-sized chunks (211 samples): the same segments, sample for sample")
    check(fails, same(real, stalled), "half-second chunks: the same")
    check(fails, same(real, long_stall), "three-second chunks: the same")
    check(fails, same(real, burst), "the whole recording in one call: the same")

    print("2. each segment holds its utterance")
    var segs = List[Seg]()
    var notes = 0
    for s in real:
        if s.start >= 0:
            segs.append(s.copy())
        else:
            notes += 1
    var pre = Int(VL_PREROLL_S * 16000.0)
    var hang = 9600                      # 0.6 s
    check(fails, len(segs) == 3, String(len(segs)) + " segments for 3 utterances")
    if len(segs) == 3:
        var starts: List[Int] = [a0, b0, c0]
        var ends: List[Int] = [a1, b1, c1]
        for k in range(3):
            var s0 = segs[k].start
            var s1 = segs[k].start + segs[k].length
            check(fails, s0 <= starts[k] and s0 >= starts[k] - pre - VL_FRAME and s1 >= ends[k] + hang - 2 * VL_FRAME and s1 <= ends[k] + hang + 3 * VL_FRAME
                  and segs[k].peak > 0.1,
                  "utterance " + String(k + 1) + ": starts " + String(starts[k] - s0) + " samples before the speech, ends "
                  + String(s1 - ends[k]) + " after it")

    print("3. what is not speech")
    check(fails, notes == 0 and len(real) == 3, "a 40 ms click opens nothing (three frames above the threshold are needed)")
    var knock = List[Int16]()
    var r5 = Rng(5)
    room(knock, 2.0, r5)
    speech(knock, 0.1, r5)         # long enough to open
    room(knock, 2.0, r5)
    var strict = G1Vad(0.6, 1.2, 10.0)      # a caller that wants 1.2 s at least
    var k1 = strict.feed(knock, True)
    check(fails, k1.opened and not k1.done and k1.note.startswith("[vad] dropped 1.0") and not strict.recording,
          "a 0.1 s knock under a 1.2 s minimum: opened, then dropped: " + k1.note)
    var faint = List[Int16]()
    room(faint, 2.0, r5)
    speech(faint, 1.0, r5, 330.0)  # 0.007 RMS: over the opening threshold (0.006), under 1.6 times it
    room(faint, 2.0, r5)
    var fv = G1Vad(0.6, 0.35, 10.0)
    var k2 = fv.feed(faint, True)
    check(fails, k2.opened and not k2.done and k2.note.startswith("[vad] dropped — peak"), "a faint sound: opened, then dropped on its peak: " + k2.note)

    print("4. a caller that may not open")
    var shut = G1Vad(0.6, 0.35, 10.0)
    var ev = shut.feed(pcm, False)
    check(fails, not ev.opened and not ev.done and not shut.recording and len(shut.ring) <= shut.ring_max and len(shut.carry) < VL_FRAME
          and shut.floor < 0.002 and shut.floor > 0.0005, "nothing opens; the pre-roll is bounded; the floor is the room's: " + String(round(shut.floor, 5)))

    print("5. busy after a segment")
    var busy = G1Vad(0.6, 0.35, 10.0)
    var head = List[Int16]()
    for i in range(b0 - 8000):
        head.append(pcm[i])         # utterance A and most of the silence after it
    var e1 = busy.feed(head, True)
    check(fails, e1.done and busy.consumed - len(busy.seg) == segs[0].start, "utterance A, as before")
    var rest_b = List[Int16]()
    for i in range(b0 - 8000, c0 - 16000):
        rest_b.append(pcm[i])       # utterance B arrives while the caller transcribes A
    var none = List[Int16]()
    var e2 = busy.feed(none, False)             # what followed A in the first read: ringed
    var e3 = busy.feed(rest_b, False)
    check(fails, not e2.done and not e3.done and not e3.opened and not busy.recording, "utterance B, spoken while the caller is busy, starts nothing")
    check(fails, e3.note == "" and e3.missed_s == 0.0 and not busy.blocked, "and without the busy flag (push-to-talk) nothing is reported")
    var rest_c = List[Int16]()
    for i in range(c0 - 16000, len(pcm)):
        rest_c.append(pcm[i])
    var e4 = busy.feed(rest_c, True)
    check(fails, e4.done and busy.consumed - len(busy.seg) == segs[2].start and len(busy.seg) == segs[2].length,
          "utterance C, once the caller is idle again, is whole and where it was")

    print("6. the echo gate")
    var gate = G1Vad(0.6, 0.35, 10.0)
    var lead = List[Int16]()
    for i in range(a0 + 8000):
        lead.append(pcm[i])         # half a second into utterance A: a segment is open
    var g1 = gate.feed(lead, True)
    check(fails, g1.opened and gate.recording and not g1.done, "a segment is open")
    var g2 = gate.skip(8000)
    check(fails, not g2.done and gate.recording and len(gate.ring) == 0, "half a second discarded: still open, no pre-roll kept")
    var g3 = gate.skip(4800)
    check(fails, g3.done and not gate.recording, "0.8 s discarded in all: the segment closed after its quiet time")
    var loud = List[Int16]()
    var r2 = Rng(7)
    speech(loud, 1.0, r2)
    var idle = G1Vad(0.6, 0.35, 10.0)
    _ = idle.feed(loud, False)
    _ = idle.skip(160)
    check(fails, len(idle.ring) == 0 and idle.above == 0 and idle.level == 0.0, "discarded audio leaves no pre-roll and no run of loud frames")

    print("7. the guards")
    var drone = List[Int16]()
    var r3 = Rng(9)
    room(drone, 1.5, r3)
    speech(drone, 12.0, r3)
    var v = G1Vad(0.6, 0.35, 10.0)
    var d = v.feed(drone, True)
    check(fails, d.done and d.over and d.seg_s > 10.0 and d.seg_s < 10.1, "a sound that never stops ends at max_seg: " + String(round(d.seg_s, 2)) + " s")
    var f = G1Vad(0.6, 0.35, 10.0)
    var quiet = List[Int16]()
    var r4 = Rng(11)
    room(quiet, 2.0, r4)
    _ = f.feed(quiet, False)
    f.open_now()
    var more = List[Int16]()
    speech(more, 1.0, r4)
    var f1 = f.feed(more, False)
    f.forced = True
    var tail = List[Int16]()
    speech(tail, 0.1, r4)
    var f2 = f.feed(tail, False)
    check(fails, not f1.done and f2.done and f.consumed - len(f.seg) <= 32000 - 0, "a forced start and a forced end: " + String(round(f2.seg_s, 2)) + " s")

    print("8. speech while the caller is busy")
    var w = G1Vad(0.6, 0.35, 10.0)
    var o1 = List[Seg]()
    play(w, pcm, 0, b0 - 8000, True, False, o1)                 # A is heard
    play(w, pcm, b0 - 8000, b1 + 24000, False, True, o1)        # busy all through B and 1.5 s more
    check(fails, len(o1) == 2 and o1[0].start == segs[0].start and o1[1].start == -1 and o1[1].peak > 1.9 and o1[1].peak < 2.1
          and o1[1].note.startswith("[vad] ") and not w.blocked,
          "B, 2 s, all of it while busy: no segment, one report: " + (o1[1].note if len(o1) == 2 else String(len(o1)) + " entries"))
    play(w, pcm, b1 + 24000, len(pcm), True, False, o1)
    check(fails, len(o1) == 3 and o1[2].start == segs[2].start and o1[2].length == segs[2].length, "C, once idle, whole and where it was")

    var m = G1Vad(0.6, 0.35, 10.0)
    var o2 = List[Seg]()
    play(m, pcm, 0, b0 - 8000, True, False, o2)
    play(m, pcm, b0 - 8000, b0 + 16000, False, True, o2)        # busy until one second into B
    var mid = len(o2)
    play(m, pcm, b0 + 16000, len(pcm), True, False, o2)         # idle from the middle of B
    check(fails, mid == 1 and len(o2) == 3 and o2[1].start == -1 and o2[1].peak > 1.9 and o2[1].peak < 2.1,
          "idle again in the middle of B: its tail opens nothing, B is reported whole: "
          + (o2[1].note if len(o2) == 3 else String(len(o2)) + " entries"))
    check(fails, len(o2) == 3 and o2[2].start == segs[2].start and o2[2].length == segs[2].length, "and C after it is whole and where it was")

    var old = G1Vad(0.6, 0.35, 10.0)
    var o3 = List[Seg]()
    play(old, pcm, 0, b0 - 8000, True, False, o3)
    play(old, pcm, b0 - 8000, b0 + 16000, False, False, o3)     # the same, the caller not saying it is busy
    play(old, pcm, b0 + 16000, len(pcm), True, False, o3)
    check(fails, len(o3) == 3 and o3[1].start >= b0 + 16000 - pre - VL_FRAME and o3[1].start < b1 and o3[1].length < 32000,
          "without the flag the tail of B is sent as an utterance: the fault, reached"
          + ((" (" + String(round(Float64(o3[1].length) / 16000.0, 2)) + " s)") if len(o3) == 3 else String("")))

    var sp = G1Vad(0.6, 0.35, 10.0)
    var o4 = List[Seg]()
    play(sp, pcm, 0, b0 - 8000, True, False, o4)
    play(sp, pcm, b0 - 8000, b0 + 16000, False, True, o4)
    var was_blocked = sp.blocked
    var sk = sp.skip(16000)                                      # the robot starts to speak over it
    check(fails, was_blocked and not sp.blocked and sk.missed_s > 0.9 and sk.missed_s < 1.1 and sk.note.startswith("[vad] "),
          "the echo gate closing over it reports what was heard of it: " + sk.note)

    print("ALL PASS" if fails == 0 else String(fails) + " FAILED")
    if fails != 0:
        exit(1)
