# +--------------------------------------------------------------------------+ #
# | The voice-activity detector — on the audio's own clock, frame by frame
# +--------------------------------------------------------------------------+ #
"""`G1Vad`: samples in, speech segments out. No microphone, no clock, no model:
the part of the voice loop that decides where an utterance starts and ends.

    var vad = G1Vad(hang_s=0.6, min_seg=0.35, max_seg=10.0)
    var ev = vad.feed(pcm, may_open=True)        # whatever the microphone gave this poll
    if ev.opened: ...                            # a segment has started
    if ev.done:   send(vad.seg)                  # a whole utterance, pre-roll included

⚠ WHY IT IS ITS OWN UNIT, AND WHY IT COUNTS SAMPLES. It used to live inside
`G1VoiceLoop.poll`, one step per POLL: the level was the RMS of whatever one
`read` returned, a segment opened after three polls above the threshold, and
it closed after 0.6 s of WALL CLOCK below it. That is right only when audio
arrives in real time, one small chunk a poll. It does not after a stall:

- the capture process was started when the loop was BUILT, the room then
  loaded for twenty seconds without a `read`, and the backlog came back faster
  than real time: 19.7 s of a recording inside 11 s of wall clock;
- one `read` then returns seconds of audio, a whole utterance is one "frame"
  averaged with the silence around it, and 0.6 s of wall-clock quiet is
  several seconds of audio.

In every headless run an utterance was lost or cut to its tail (« Je m'appelle
Denis. » heard as « Denis. », a 0.99 s segment peaking at 0.011 against 0.26):
docs/ROBOT_BRAIN_PLAN.md §11.4 to §14.7.

So: audio is cut into frames of `VL_FRAME` samples (20 ms) wherever the chunk
boundaries fall, every threshold is applied per frame, and every duration
(the opening run, the closing quiet, the segment's length) is a count of
SAMPLES. The same samples give the same segments however they are delivered
(`tests/robots/test_g1_vad.mojo` holds it to that, sample for sample), and the
voice loop now opens its capture at the first poll, not when it is built.

`feed` stops at the end of a finished segment and keeps what follows for the
next call: the caller decides, with `may_open`, whether the audio after an
utterance may start another (the voice loop says no while a transcription or a
decision is in flight, as before).

AN EARLY PAUSE (`early_s`, 0 = off). A segment closes after `hang_s` of quiet
(0.6 s): waiting that long is how a pause inside a sentence is told from its
end. With `early_s` (0.3 s in the room demo) the detector also says when a
segment it would keep has been quiet that long: `paused` turns True and
`pause_seq` counts it; speech again turns `paused` back to False. A caller can
start transcribing what is in hand at the pause and keep it when the segment
does close on that same pause (`pause_seq` unchanged, still `paused`), or drop
it when the person goes on (docs/NN_MODEL_PORTS_PLAN.md §12.8.7, L3). State,
not an event: the same samples give the same `paused` / `pause_seq` however
they are chunked, and a caller reading them after each `feed` sees where the
audio it was given has got to.

⚠ SPEECH WHILE THE CALLER IS BUSY IS LOST ON PURPOSE, BUT NEVER SILENTLY, AND
NEVER IN HALVES (`busy`). The count above was not the whole fault. With the
detector exact, a replayed recording still lost its second utterance: the first
one had been spoken while the models were still loading, its transcription
came back 6 s late, and the second arrived while the loop was deciding the
first. Nothing was printed. And when the loop came back to idle in the MIDDLE
of it, the detector opened on what was left: « Je m'appelle Denis. » heard as
« Denis. », a 0.99 s segment peaking at 0.013. So speech that begins while the
caller is busy now BLOCKS the detector until it has ended (the same quiet that
ends a segment), and its end is reported: `[vad] 1.9 s of speech while busy —
not heard` (`G1VadEvent.missed_s`).
"""

comptime VL_FRAME: Int = 320
"""Samples in one detector frame: 20 ms at 16 kHz. With `VL_OPEN_FRAMES` = 3
a segment opens on 60 ms above the threshold, what three polls of a 50 Hz
render loop were."""

comptime VL_PREROLL_S: Float64 = 0.35
"""Audio kept before the detector opens, so the first syllable survives."""
comptime VL_DRAIN_S: Float64 = 0.35

# ── the voice-activity detector, every constant measured ──────────────────
comptime VL_OPEN_MULT: Float64 = 5.0
comptime VL_CLOSE_MULT: Float64 = 2.5
comptime VL_OPEN_MIN: Float64 = 0.0060
comptime VL_OPEN_FRAMES: Int = 3
comptime VL_MIN_PEAK_MULT: Float64 = 1.6
comptime VL_FLOOR_MIN: Float64 = 0.00005


# ── the detector's arithmetic, as functions so it can be GATED ────────────
# ⚠ THIS WAS INLINE IN THE LOOP AND THEREFORE UNTESTABLE, which is the wrong
# place for the three lines with the worst history in this file. Both of
# §12.55's detector bugs live here.


def g1_vad_open_at(floor: Float64) -> Float64:
    """The level a segment opens at. ⚠ FLOORED, because `floor * 5` in a very
    quiet room is below the microphone's own noise and the detector would open
    on nothing."""
    var v = floor * VL_OPEN_MULT
    return v if v > VL_OPEN_MIN else VL_OPEN_MIN


def g1_vad_close_at(floor: Float64) -> Float64:
    """The level a segment closes below.

    ⚠ IT NEEDS A FLOOR OF ITS OWN, and this is the bug that cost a session.
    It was `floor * 2.5` alone, and the floor is an average over QUIET frames
    — so in a room whose between-word noise sits above that, a segment opens
    and NEVER CLOSES: one ran the full 10 s max-segment guard for a 3 s
    question, and another recorded 5.99 s for the word "you". Tying it to the
    opening threshold keeps the two in proportion whatever the room is doing.
    """
    var v = floor * VL_CLOSE_MULT
    var lo = VL_OPEN_MIN * 0.5
    return v if v > lo else lo


def g1_vad_floor_step(floor: Float64, level: Float64) -> Float64:
    """One EMA step of the tracked quiet level.

    ⚠ THE FLOOR IS TYPICAL QUIET, NOT THE QUIETEST INSTANT. Tracking the
    minimum put `close_at` below ordinary room noise, which is the same
    never-closing failure from the other direction. ⚠ And the CALLER must only
    call this on frames BELOW `g1_vad_open_at`, or speech raises the floor it
    is measured against and the detector deafens itself mid-sentence.
    """
    var f = floor * 0.98 + level * 0.02
    return f if f > VL_FLOOR_MIN else VL_FLOOR_MIN


@fieldwise_init
struct G1VadEvent(Copyable, Movable):
    var opened: Bool
    """A segment opened during this call."""
    var done: Bool
    """A segment is complete: it is `G1Vad.seg`, pre-roll included."""
    var over: Bool
    """It ended on the max-segment guard, not on silence."""
    var seg_s: Float64
    var note: String
    """`[vad] dropped ...` when a segment was discarded, `[vad] ... of speech
    while busy` when one was never started; "" otherwise."""
    var missed_s: Float64
    """Above 0: speech that began while the caller was busy has just ended,
    this long, and none of it was kept."""

    @staticmethod
    def none() -> G1VadEvent:
        return G1VadEvent(False, False, False, 0.0, String(""), 0.0)


struct G1Vad(Movable):
    var floor: Float64
    """The tracked quiet level (`g1_vad_floor_step`)."""
    var level: Float64
    """The last frame's RMS level, in [0, 1]."""
    var peak: Float64
    """The loudest frame of the segment in hand (or since the last one)."""
    var above: Int
    """Consecutive frames above the opening threshold."""
    var ring: List[Int16]
    """The last `ring_max` samples while no segment is open: the pre-roll."""
    var ring_max: Int
    var seg: List[Int16]
    """The segment in hand, or the one just finished."""
    var recording: Bool
    var quiet: Int
    """Samples since a frame was last above the closing threshold."""
    var forced: Bool
    """End the segment in hand at the next frame (`G1VoiceLoop.force`)."""
    var carry: List[Int16]
    """Samples fed and not yet processed: less than a frame, or what followed
    a finished segment."""
    var consumed: Int
    """Samples processed since the start: a finished segment ends here."""
    var blocked: Bool
    """Speech began while the caller was busy: nothing opens until it ends."""
    var missed: Int
    """Samples of that speech so far, its inner pauses included."""
    var calm: Int
    """Samples since its last frame above the closing threshold."""
    var hang_s: Float64
    var min_seg: Float64
    var max_seg: Float64
    var early_s: Float64
    """Quiet that reports a pause (`paused`); 0 = never (the module note)."""
    var paused: Bool
    """The segment in hand has been quiet for `early_s` and would be kept if it
    ended now; False again when speech resumes, or a new segment opens."""
    var pause_seq: Int
    """Pauses reported so far (one per quiet stretch)."""

    def __init__(out self, hang_s: Float64, min_seg: Float64, max_seg: Float64):
        self.floor = 0.0010
        self.level = 0.0
        self.peak = 0.0
        self.above = 0
        self.ring = List[Int16]()
        self.ring_max = Int(VL_PREROLL_S * 16000.0)
        self.seg = List[Int16]()
        self.recording = False
        self.quiet = 0
        self.forced = False
        self.carry = List[Int16]()
        self.consumed = 0
        self.blocked = False
        self.missed = 0
        self.calm = 0
        self.hang_s = hang_s
        self.min_seg = min_seg
        self.max_seg = max_seg
        self.early_s = 0.0
        self.paused = False
        self.pause_seq = 0

    def restart(mut self):
        """The microphone was reopened: nothing is in hand, and the floor
        starts over (it described a room that may have changed)."""
        self.ring = List[Int16]()
        self.carry = List[Int16]()
        self.seg = List[Int16]()
        self.recording = False
        self.level = 0.0
        self.peak = 0.0
        self.floor = 0.0010
        self.above = 0
        self.quiet = 0
        self.forced = False
        self.blocked = False
        self.missed = 0
        self.calm = 0
        self.paused = False

    def open_now(mut self):
        """Start a segment from the pre-roll, whatever the level (TAB)."""
        self.seg = self.ring.copy()
        self.ring = List[Int16]()
        self.peak = self.level
        self.quiet = 0
        self.forced = False
        self.recording = True
        self.paused = False

    def _would_keep(self, open_at: Float64) -> Bool:
        """The segment in hand would be kept if it ended now (`_finish`'s rules)."""
        var seg_s = Float64(len(self.seg)) / 16000.0
        return seg_s > VL_DRAIN_S and seg_s >= self.min_seg and self.peak >= open_at * VL_MIN_PEAK_MULT

    def _finish(mut self, mut ev: G1VadEvent, open_at: Float64) -> Bool:
        """End the segment in hand if it is due. True when it ended (kept, in
        `seg`, or dropped with a note)."""
        var seg_s = Float64(len(self.seg)) / 16000.0
        # ⚠ THREE WAYS TO END, and the last two are not optional. Silence is
        # the normal one. A segment that never falls quiet (a fan, a
        # conversation across the room) would otherwise record for ever; a
        # forced end is what TAB is for.
        var over = seg_s > self.max_seg
        var due = self.forced or over or (Float64(self.quiet) / 16000.0 > self.hang_s)
        if not (due and seg_s > VL_DRAIN_S):
            return False
        self.forced = False
        self.recording = False
        # ⚠ a cough, a chair, a door. Below `min_seg` it is not speech, and
        # sending it costs a transcription to be told so.
        if seg_s < self.min_seg:
            ev.note = (String("[vad] dropped ") + String(round(seg_s, 2)) + String(" s — under ")
                       + String(round(self.min_seg, 2)))
        elif self.peak < open_at * VL_MIN_PEAK_MULT:
            ev.note = (String("[vad] dropped — peak ") + String(round(self.peak, 4)) + String(" under ")
                       + String(round(open_at * VL_MIN_PEAK_MULT, 4)))
        else:
            ev.done = True
            ev.over = over
            ev.seg_s = seg_s
        return True

    def _release(mut self, mut ev: G1VadEvent):
        """The speech the caller was too busy for has ended: say so."""
        ev.missed_s = Float64(self.missed) / 16000.0
        var line = String("[vad] ") + String(round(ev.missed_s, 2)) + String(" s of speech while busy — not heard")
        ev.note = line if ev.note == "" else ev.note + String(" | ") + line
        self.blocked = False
        self.missed = 0
        self.calm = 0

    def skip(mut self, samples: Int) -> G1VadEvent:
        """`samples` were read and DISCARDED (the robot is speaking: the echo
        gate). Nothing of them is levelled or kept, and the pre-roll is
        dropped with them, or it would hand the recogniser the tail of the
        robot's own sentence as the first syllable of the next one. A segment
        in hand goes on ageing: it closes after its quiet time as it did when
        that time was the wall clock's."""
        var ev = G1VadEvent.none()
        self.level = 0.0
        self.above = 0
        self.ring = List[Int16]()
        self.carry = List[Int16]()
        self.consumed += samples
        if self.blocked:
            # the robot now speaks over it: whatever is left cannot be told
            # from the robot's own voice
            self._release(ev)
        if self.recording:
            self.quiet += samples
            _ = self._finish(ev, g1_vad_open_at(self.floor))
        return ev^

    def feed(mut self, ref pcm: List[Int16], may_open: Bool, busy: Bool = False) -> G1VadEvent:
        """Take `pcm` (any length, even none) and process whole frames, up to
        the end of ONE finished segment; what follows stays in `carry`.
        `may_open`: a segment may start (the caller is idle). `busy`: the
        caller is working on the previous utterance, so speech that begins now
        is lost WHOLE and reported at its end (the module note). Leave it
        False when segments are not opened by level at all (push-to-talk)."""
        var ev = G1VadEvent.none()
        for x in pcm:
            self.carry.append(x)
        var n = len(self.carry)
        var i = 0
        while n - i >= VL_FRAME and not ev.done:
            var acc = 0.0
            for k in range(VL_FRAME):
                var v = Float64(self.carry[i + k]) / 32768.0
                acc += v * v
            var lv = (acc / Float64(VL_FRAME)) ** 0.5
            self.level = lv
            if lv > self.peak:
                self.peak = lv
            var open_at = g1_vad_open_at(self.floor)
            var close_at = g1_vad_close_at(self.floor)
            if self.recording:
                for k in range(VL_FRAME):
                    self.seg.append(self.carry[i + k])
            else:
                for k in range(VL_FRAME):
                    self.ring.append(self.carry[i + k])
                if len(self.ring) > self.ring_max:
                    var keep = List[Int16](capacity=self.ring_max)
                    for k in range(len(self.ring) - self.ring_max, len(self.ring)):
                        keep.append(self.ring[k])
                    self.ring = keep^
                # ⚠ THE FLOOR IS TYPICAL QUIET, NOT THE QUIETEST INSTANT, and
                # only frames BELOW the open threshold count, or speech raises
                # the floor it is measured against and the detector deafens
                # itself mid-sentence (`g1_vad_floor_step`).
                if lv < open_at:
                    self.floor = g1_vad_floor_step(self.floor, lv)
            if lv > open_at:
                self.above += 1
            else:
                self.above = 0
            i += VL_FRAME
            self.consumed += VL_FRAME
            if not self.recording:
                if self.blocked:
                    if lv > close_at:
                        self.missed += VL_FRAME + self.calm
                        self.calm = 0
                    else:
                        self.calm += VL_FRAME
                        if Float64(self.calm) / 16000.0 > self.hang_s:
                            self._release(ev)
                elif busy and self.above >= VL_OPEN_FRAMES:
                    self.blocked = True
                    self.missed = VL_OPEN_FRAMES * VL_FRAME
                    self.calm = 0
            if not self.recording and may_open and not self.blocked and self.above >= VL_OPEN_FRAMES:
                # the pre-roll already holds this frame
                self.open_now()
                self.peak = lv
                ev.opened = True
            if self.recording:
                if lv > close_at:
                    self.quiet = 0
                    self.paused = False     # the person goes on
                else:
                    self.quiet += VL_FRAME
                    if (self.early_s > 0.0 and not self.paused and Float64(self.quiet) / 16000.0 > self.early_s
                            and self._would_keep(open_at)):
                        self.paused = True
                        self.pause_seq += 1
                _ = self._finish(ev, open_at)
        if i > 0:
            var rest = List[Int16](capacity=n - i)
            for k in range(i, n):
                rest.append(self.carry[k])
            self.carry = rest^
        return ev^
