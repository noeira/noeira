"""The channel is polled 50 times a second by a render loop, so it gets a gate.

    pixi run mojo run -I . tests/robots/test_g1_command_channel.mojo

WHY THIS EXISTS
===============
Two properties matter and both are invisible when broken.

**It must never raise.** The file is absent until a writer starts, and can be
caught mid-rename. A `poll` that threw would take the renderer — and with it
the physics step — down because a text file was briefly missing.

**`seq` must gate.** Polling is idempotent only if an unchanged file costs a
comparison; if `poll` returned `fresh` every frame, a command would be
re-applied 50 times a second and every blend would restart on the frame it
began, so the robot would never finish a transition.
"""

from noeira.io.fileio import write_text_atomic, remove_file
from noeira.envs.robots.g1_command_channel import (
    G1CommandChannel, g1_channel_write,
)
from noeira.core.bytes import string_from_bytes
from noeira.io.fileio import read_file_bytes


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
    print("=== g1 command channel ===")
    var path = String("/tmp/noeira_test_chan")

    # ── an ABSENT file is normal, not an error ────────────────────────
    try:
        remove_file(path)
    except:
        pass
    var ch = G1CommandChannel(path)
    var m = ch.poll()
    t.truth(not m.fresh, "an absent channel polls as not-fresh and does not raise")

    # ── a round trip ──────────────────────────────────────────────────
    g1_channel_write(path, 1, String("squat"), 25)
    m = ch.poll()
    t.truth(m.fresh, "a written command reads as fresh")
    t.truth(m.cmd == "squat", String("the command survives the round trip (") + m.cmd + String(")"))
    t.truth(m.seq == 1, "the seq survives")
    t.truth(m.blend == 25, "the blend survives")

    # ── ⚠ THE SAME FILE MUST NOT READ TWICE ───────────────────────────
    # Without this, a blend restarts every frame and never completes.
    m = ch.poll()
    t.truth(not m.fresh, "re-polling the UNCHANGED file is not fresh")
    m = ch.poll()
    t.truth(not m.fresh, "...and still not fresh on a third poll")

    # ── a new seq reopens the gate ────────────────────────────────────
    g1_channel_write(path, 2, String("walk"), 10)
    m = ch.poll()
    t.truth(m.fresh and m.cmd == "walk", "a higher seq is fresh again")

    # ⚠ THE SAME COMMAND WITH A NEW SEQ MUST FIRE. A writer that re-sends
    # `stand` means it — deduplicating on the command NAME would swallow a
    # repeated stop.
    g1_channel_write(path, 3, String("walk"), 10)
    m = ch.poll()
    t.truth(m.fresh, "the SAME command with a new seq fires again")

    # ── a stale or replayed seq is ignored ────────────────────────────
    g1_channel_write(path, 2, String("squat"), 10)
    m = ch.poll()
    t.truth(not m.fresh, "a LOWER seq is ignored (a stale writer cannot rewind)")

    # ── malformed input must not raise ────────────────────────────────
    write_text_atomic(path, String("cmd squ"))          # a torn write
    m = ch.poll()
    t.truth(not m.fresh, "a torn file (no seq) is not fresh and does not raise")
    write_text_atomic(path, String("seq 99\n"))         # seq, no cmd
    m = ch.poll()
    t.truth(not m.fresh, "a seq with no command is not fresh")
    write_text_atomic(path, String(""))
    m = ch.poll()
    t.truth(not m.fresh, "an empty file is not fresh")
    write_text_atomic(path, String("garbage\nmore garbage\n"))
    m = ch.poll()
    t.truth(not m.fresh, "unparseable content is not fresh")

    # the gate must still open afterwards — a torn read must not poison it
    g1_channel_write(path, 100, String("stand"), 5)
    m = ch.poll()
    t.truth(m.fresh and m.cmd == "stand", "the channel recovers after malformed input")

    # ── the ack ───────────────────────────────────────────────────────
    ch.ack(100, String("stand"), True)
    var ack = string_from_bytes(read_file_bytes(path + String(".ack")))
    t.truth(ack.find("status ok") >= 0, "an accepted command acks `ok`")
    ch.ack(101, String("fly"), False)
    ack = string_from_bytes(read_file_bytes(path + String(".ack")))
    t.truth(ack.find("status unknown") >= 0, "a rejected command acks `unknown`")
    t.truth(ack.find("fly") >= 0, "the ack names the command it refused")

    try:
        remove_file(path)
        remove_file(path + String(".ack"))
    except:
        pass
    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_g1_command_channel: " + String(t.fails) + " failed")
