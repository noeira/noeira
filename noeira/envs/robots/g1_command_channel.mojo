# +--------------------------------------------------------------------------+ #
# | The command channel — a file the render loop polls
# +--------------------------------------------------------------------------+ #
"""How a command reaches the robot without stalling it.

    writer (any process, any language):   g1_channel_write(path, seq, "squat")
    viewer (the 50 Hz loop):              var m = chan.poll()

WHY A FILE AND NOT A WIDGET
===========================
Not a preference — a requirement. The things that will drive this robot are
slow: speech-to-text is ~1.1 s and a constrained-readout model ~0.3 s
(`noeira/ai/`, measured). The render loop has **20 ms** a frame. A call made
inline freezes it for roughly seventy frames, and a frozen viewer is also a
frozen physics step.

A file breaks that coupling completely. The model process blocks as much as
it likes and writes one short line; the loop does an open/read/close of a few
dozen bytes and moves on. It also means chat, speech, a language model and a
shell one-liner are all the same thing — a writer — and none of them needs to
link against the renderer or be written in Mojo.

`docs/SYSTEM_ONE_ASSESSMENT.md` §5.1 rules out an API call in the control
loop. With this, the API is UPSTREAM of the loop rather than inside it.

THE FORMAT
==========
Whole file, rewritten each time, three lines at most:

    seq 7
    cmd squat
    blend 25          # optional: frames to slerp over, 0 snaps

⚠ WRITERS MUST WRITE ATOMICALLY (`write_text_atomic`, which writes a
temporary and renames). A plain rewrite lets the reader catch the file
mid-write and see `cmd squ` — which resolves to no bank entry, so the command
is silently dropped rather than corrupting anything, but it is dropped.

⚠ `seq` MUST INCREASE. The reader ignores anything it has already seen, which
is what makes polling idempotent: re-reading an unchanged file costs a
comparison. Sending the same command twice needs a new `seq`, and that is
deliberate — a writer that re-sends `stop` wants it to take effect.

THE ACK
=======
The reader writes `<path>.ack` back: the seq it consumed, the command, and
whether it resolved. A voice layer needs that last field — the bank returns
-1 for a name that was rejected at build time or never defined, and the
honest response is to say so, not to run the nearest-looking entry.
"""

from noeira.io.fileio import read_file_bytes, write_text_atomic
from noeira.core.bytes import string_from_bytes


struct G1ChannelMsg(Copyable, Movable):
    var seq: Int
    var cmd: String
    var blend: Int
    var fresh: Bool
    """False when there was nothing new, the file was absent, or it did not
    parse. A caller keeps doing whatever it was doing."""

    def __init__(out self):
        self.seq = -1
        self.cmd = String("")
        self.blend = 0
        self.fresh = False


struct G1CommandChannel(Movable):
    var path: String
    var last_seq: Int

    def __init__(out self, path: String):
        self.path = path
        self.last_seq = -1

    def __init__(out self, *, deinit move: Self):
        self.path = move.path^
        self.last_seq = move.last_seq

    def poll(mut self) -> G1ChannelMsg:
        """Read the channel. Never raises.

        ⚠ AN ABSENT OR HALF-WRITTEN FILE IS NORMAL, not an error: the channel
        is optional, the writer may not have started, and a non-atomic writer
        can be caught mid-rename. Every one of those returns `fresh = False`
        and the loop carries on. A `raise` here would take the renderer down
        because a text file was briefly missing.
        """
        var m = G1ChannelMsg()
        try:
            var raw = read_file_bytes(self.path)
            var txt = string_from_bytes(raw)
            var lines = txt.split("\n")
            var seq = -1
            var cmd = String("")
            var blend = 0
            for i in range(len(lines)):
                var l = String(lines[i])
                if l.byte_length() == 0 or l.startswith("#"):
                    continue
                var p = l.split(" ")
                if len(p) < 2:
                    continue
                if l.startswith("seq "):
                    seq = atol(String(p[1]))
                elif l.startswith("cmd "):
                    cmd = String(p[1])
                elif l.startswith("blend "):
                    blend = atol(String(p[1]))
            if seq <= self.last_seq or cmd == "":
                return m^
            self.last_seq = seq
            m.seq = seq
            m.cmd = cmd
            m.blend = blend
            m.fresh = True
        except:
            # absent, unreadable, or caught mid-write — all the same answer
            pass
        return m^

    def ack(self, seq: Int, cmd: String, ok: Bool):
        """Tell the writer what happened. Never raises — a demo does not stop
        because a status file could not be written."""
        try:
            var s = String("seq ") + String(seq) + String("\n")
            s += String("cmd ") + cmd + String("\n")
            s += String("status ") + (String("ok") if ok else String("unknown")) + String("\n")
            write_text_atomic(self.path + String(".ack"), s)
        except:
            pass


def g1_channel_seq(path: String) -> Int:
    """The seq currently in the channel, or -1.

    ⚠ A WRITER MUST ADVANCE PAST THIS, not start from 1. Two writers (a voice
    process and a shell script, say) sharing a channel would otherwise fight:
    the second one's `seq 1` is below the first's `seq 7` and the reader
    ignores it, so the command is silently dropped.
    """
    try:
        var txt = string_from_bytes(read_file_bytes(path))
        var lines = txt.split("\n")
        for i in range(len(lines)):
            var l = String(lines[i])
            if l.startswith("seq "):
                var p = l.split(" ")
                if len(p) >= 2:
                    return atol(String(p[1]))
    except:
        pass
    return -1


def g1_channel_read_ack(path: String) -> G1ChannelMsg:
    """The reader's last acknowledgement. `cmd` is the command it saw and
    `fresh` is True when the status was `ok` — so a writer can tell an
    accepted command from a rejected one without guessing."""
    var m = G1ChannelMsg()
    try:
        var txt = string_from_bytes(read_file_bytes(path + String(".ack")))
        var lines = txt.split("\n")
        for i in range(len(lines)):
            var l = String(lines[i])
            var p = l.split(" ")
            if len(p) < 2:
                continue
            if l.startswith("seq "):
                m.seq = atol(String(p[1]))
            elif l.startswith("cmd "):
                m.cmd = String(p[1])
            elif l.startswith("status "):
                m.fresh = String(p[1]) == "ok"
    except:
        pass
    return m^


def g1_channel_write(path: String, seq: Int, cmd: String, blend: Int = 25) raises:
    """Send one command. Atomic, so a reader polling at 50 Hz cannot see a
    partial line."""
    var s = String("seq ") + String(seq) + String("\n")
    s += String("cmd ") + cmd + String("\n")
    s += String("blend ") + String(blend) + String("\n")
    write_text_atomic(path, s)
