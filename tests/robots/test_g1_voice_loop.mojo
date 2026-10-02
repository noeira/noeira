"""The voice loop's pure parts — a gate, because two of them cost a session.

    pixi run mojo run -I . tests/robots/test_g1_voice_loop.mojo

WHY THIS EXISTS
===============
`G1VoiceLoop` owns a microphone, so most of it cannot be unit-tested. The
detector's ARITHMETIC can, and it is the part with the worst history:

**A segment that opened and never closed.** `close_at` was `floor * 2.5` with
no floor of its own, and the tracked floor is an average over QUIET frames —
so in a room whose between-word noise sits above that, nothing ever falls
below the closing threshold. Measured: 5.99 s recorded for the word "you",
7.06 s for "Cool. Run.", both ending on the 10 s max-segment guard rather
than on silence (§12.55).

**A floor that tracked the minimum rather than typical quiet**, which put the
same threshold below ordinary room noise — the identical failure reached from
the other direction.

So the invariant that matters is `close_at < open_at` AT EVERY FLOOR: if they
cross, a segment either cannot open or cannot close, and both look like the
microphone being broken. It is asserted here across six orders of magnitude.

The deadlines are pinned too, because they are derived from measured latencies
(Whisper on the HF endpoint has returned 0.71 to 5.41 s across §12.55-12.63)
and a future edit that tightens one below the tail it was chosen for would
reintroduce §12.60's "GAVE UP" on a call that was merely slow.

Nothing here opens a device or touches the network.
"""

from noeira.envs.robots.g1_voice_loop import (
    G1VoiceConfig, G1VoiceEvent,
    g1_vad_open_at, g1_vad_close_at, g1_vad_floor_step,
    VL_NONE, VL_COMMAND, VL_SPEC, VL_WORLD, VL_TALK, VL_REFUSED,
    VL_ST_IDLE, VL_ST_REC, VL_ST_STT, VL_ST_JEV, VL_ST_SPEC, VL_ST_CHAT,
    VL_OPEN_MULT, VL_CLOSE_MULT, VL_OPEN_MIN, VL_FLOOR_MIN,
    VL_OPEN_FRAMES, VL_MIN_PEAK_MULT, VL_ECHO_TAIL_S,
    VL_STT_DEADLINE_S, VL_JEV_DEADLINE_S, VL_CHAT_DEADLINE_S,
)


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def main() raises:
    var t = Tally()

    # ── the invariant that cost a session ─────────────────────────────
    print("-- close_at < open_at at every floor")
    var floors = List[Float64]()
    floors.append(0.0)
    floors.append(1e-6)
    floors.append(VL_FLOOR_MIN)
    floors.append(0.0001)
    floors.append(0.0010)
    floors.append(0.0060)
    floors.append(0.05)
    floors.append(0.5)
    var crossed = 0
    for i in range(len(floors)):
        var o = g1_vad_open_at(floors[i])
        var c = g1_vad_close_at(floors[i])
        if c >= o:
            crossed += 1
            print("    CROSSED at floor", floors[i], ": open", o, "close", c)
    t.truth(crossed == 0,
            "the thresholds never cross, over six orders of magnitude")

    # ⚠ and the specific regression: a very low floor must NOT produce a
    # close threshold of nearly zero, which is what never closed.
    t.truth(g1_vad_close_at(0.0) >= VL_OPEN_MIN * 0.5,
            "a zero floor still gives a closing threshold above zero")
    t.truth(g1_vad_close_at(1e-9) >= VL_OPEN_MIN * 0.5,
            "and so does an absurdly quiet room")
    t.truth(g1_vad_open_at(0.0) >= VL_OPEN_MIN,
            "a zero floor cannot open the detector on nothing")

    # ── the thresholds track the floor when it is loud enough ─────────
    print("-- and they scale with the room above the floors")
    var loud = 0.05
    t.truth(g1_vad_open_at(loud) > VL_OPEN_MIN,
            "a loud room raises the opening threshold above its minimum")
    var ratio = g1_vad_open_at(loud) / g1_vad_close_at(loud)
    t.truth(ratio > 1.9 and ratio < 2.1,
            "open/close stay in proportion (5.0 / 2.5 = 2.0, got "
            + String(ratio) + ")")

    # ── the floor is an EMA, clamped, and moves slowly ────────────────
    print("-- the floor step")
    t.truth(g1_vad_floor_step(0.0, 0.0) == VL_FLOOR_MIN,
            "the floor is clamped above zero")
    var f = 0.0010
    var up = g1_vad_floor_step(f, 0.0100)
    t.truth(up > f and up < f + 0.0002,
            "a loud frame moves the floor by ~2 %, not to the level")
    var dn = g1_vad_floor_step(f, 0.0)
    t.truth(dn < f and dn > f * 0.97,
            "a silent frame lowers it by ~2 %")
    # ⚠ 100 loud frames must still not take the floor to the level — the
    # detector would otherwise deafen itself over a long sentence. This is
    # why the CALLER gates the step on `level < open_at`.
    var g = 0.0010
    for _ in range(100):
        g = g1_vad_floor_step(g, 0.0100)
    t.truth(g < 0.0100,
            "100 loud frames do not reach the level (" + String(g) + ")")

    # ── the deadlines, pinned to their measured origins ───────────────
    print("-- deadlines")
    t.truth(VL_STT_DEADLINE_S >= 15.0,
            "the STT budget is at least 3x the slowest Whisper seen (5.41 s)")
    t.truth(VL_JEV_DEADLINE_S >= 6.0,
            "the Jev budget is at least 10x the slowest decision (0.61 s)")
    t.truth(VL_CHAT_DEADLINE_S > VL_JEV_DEADLINE_S,
            "a generative reply gets longer than a constrained readout")
    t.truth(VL_STT_DEADLINE_S > VL_JEV_DEADLINE_S,
            "transcription gets longer than a decision")
    t.truth(VL_ECHO_TAIL_S > 0.0 and VL_ECHO_TAIL_S < 2.0,
            "the echo tail is a buffer drain, not an utterance estimate")

    # ── the event kinds are distinct, and `none` is inert ─────────────
    print("-- events")
    var kinds = List[Int]()
    kinds.append(VL_NONE)
    kinds.append(VL_COMMAND)
    kinds.append(VL_SPEC)
    kinds.append(VL_WORLD)
    kinds.append(VL_TALK)
    kinds.append(VL_REFUSED)
    var dup = 0
    for i in range(len(kinds)):
        for j in range(i + 1, len(kinds)):
            if kinds[i] == kinds[j]:
                dup += 1
    t.truth(dup == 0, "the six event kinds are distinct")
    var e = G1VoiceEvent.none()
    t.truth(e.kind == VL_NONE, "`none()` is VL_NONE")
    t.truth(e.name == "" and e.destination == "" and len(e.z) == 0,
            "`none()` carries nothing a caller could act on")
    # ⚠ VL_WORLD must never arrive WITH a command name — `g1_decide` clears it
    # so a stale bank row cannot race the planner. The event's default must
    # not reintroduce one.
    t.truth(e.text == "" and e.conf == 0.0, "`none()` has no confidence")

    # ── the states are distinct ───────────────────────────────────────
    var sts = List[Int]()
    sts.append(VL_ST_IDLE)
    sts.append(VL_ST_REC)
    sts.append(VL_ST_STT)
    sts.append(VL_ST_JEV)
    sts.append(VL_ST_SPEC)
    sts.append(VL_ST_CHAT)
    var sdup = 0
    for i in range(len(sts)):
        for j in range(i + 1, len(sts)):
            if sts[i] == sts[j]:
                sdup += 1
    t.truth(sdup == 0, "the six states are distinct")

    # ── the config's defaults are the measured ones ───────────────────
    print("-- config defaults")
    var c = G1VoiceConfig()
    t.truth(c.stt_dl == VL_STT_DEADLINE_S and c.jev_dl == VL_JEV_DEADLINE_S,
            "the config defaults to the module's measured deadlines")
    t.truth(c.max_none == 0.25,
            "`max_none` is §12.54's measured abstention bar")
    t.truth(c.min_top == 0.35, "`min_top` is the second, lower bar")
    t.truth(c.max_seg == 10.0,
            "the max-segment guard is the one that caught the never-closing bug")
    t.truth(c.min_seg > 0.0 and c.min_seg < c.max_seg,
            "min_seg is positive and below max_seg")
    t.truth(c.vad, "hands-free is the default")
    t.truth(c.pool_path == "",
            "the cache-miss path is OPT-IN — a 70 MB read is not a default")
    t.truth(len(c.destinations) == 0,
            "no destinations by default, so the request is byte-identical to "
            "a roomless one")
    t.truth(c.stt_spec == "hf", "the recogniser defaults to HF")
    # ⚠ the vocabulary must be empty by default, because on the HF backend a
    # non-empty prompt RAISES — a default that crashed the first transcription
    # is exactly what §12.56 shipped by accident.
    t.truth(c.vocab == "",
            "no vocabulary prompt by default (HF raises on one)")

    t.truth(VL_OPEN_FRAMES >= 2,
            "a single loud frame cannot open a segment")
    t.truth(VL_MIN_PEAK_MULT > 1.0,
            "a segment must get louder than the threshold that opened it")
    t.truth(VL_OPEN_MULT > VL_CLOSE_MULT,
            "the opening multiplier exceeds the closing one, or it would "
            "close as soon as it opened")

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_g1_voice_loop: " + String(t.fails) + " failed")
