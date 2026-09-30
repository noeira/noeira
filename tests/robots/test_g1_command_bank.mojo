"""The bank reader decides what `z` the robot is driven with, so it gets a gate.

    pixi run mojo run -I . tests/robots/test_g1_command_bank.mojo

WHY THIS EXISTS
===============
`G1CommandBank.load` turns a text file into the 256 floats that condition the
policy. Every way it can go wrong is SILENT: any 256 floats project onto the
radius-16 sphere and produce a robot that does something plausible. A `z` row
one value short would slide every later command's vector by one, and each
command would drive a different behaviour than its name — with no error, no
NaN, and a demo that looks merely disappointing rather than broken.

So the width check is the point of the reader, and this is the gate on it.
The fixture is written here rather than read from `g1_command_bank.txt`,
because a test that reads the real bank would pass on an empty file.
"""

from noeira.io.fileio import write_text_atomic, remove_file
from noeira.envs.robots.g1_command_bank import G1CommandBank


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


def _row(v: Float64, n: Int) -> String:
    var s = String("z")
    for _ in range(n):
        s += String(" ") + String(v)
    return s + String("\n")


def _bank(dim: Int, w1: Int, w2: Int) -> String:
    var s = String("# g1 command bank v1\n# ckpt test\ncount 2 ") + String(dim) + String("\n")
    s += String("name squat\ngroup posture\n")
    s += String("term 0 2 0.4 0.62  # body_height band  <- GOAL\n")
    s += String("ess 140\nhard 1.0 1.0 cem\n")
    s += String("achieved 0.513 0.86 0.68 0.67 0.1 0.1 0.05 0.05 0.999 0.0 0.0 0.0 0.0 0.0\n")
    s += _row(0.5, w1)
    s += String("name walk\ngroup locomotion\n")
    s += String("term 9 2 0.5 1.4  # body_speed_forward band  <- GOAL\n")
    s += String("ess 16756\nhard 0.0 1.0 cem\n")
    s += String("achieved 0.765 1.10 0.75 0.75 0.1 0.1 0.05 0.05 0.991 1.023 0.0 1.03 0.0 0.0\n")
    s += _row(-0.25, w2)
    return s


def main() raises:
    var t = Tally()
    print("=== g1 command bank reader ===")
    var path = String("/tmp/noeira_test_bank.txt")

    # ── a well-formed bank ────────────────────────────────────────────
    write_text_atomic(path, _bank(256, 256, 256))
    var b = G1CommandBank.load(path)
    t.truth(b.count() == 2, String("two entries loaded (") + String(b.count()) + String(")"))
    t.truth(b.dim == 256, String("dim read from the header (") + String(b.dim) + String(")"))
    t.truth(b.find("squat") == 0, "find returns the index of the first entry")
    t.truth(b.find("walk") == 1, "find returns the index of the second entry")
    t.truth(b.name_at(1) == "walk", "name_at agrees with find")
    t.truth(b.group_at(0) == "posture", "the group is carried through")
    t.truth(b.hold_at(1) == 1.0, "the SHIPPED hold is read, not the zero-shot one")

    # ⚠ the two rows must not be confused for one another — that is exactly
    # what an off-by-one in the row stride produces, and it is invisible.
    t.truth(b.z_at(0, 0) == 0.5 and b.z_at(0, 255) == 0.5, "entry 0's z is its own")
    t.truth(b.z_at(1, 0) == -0.25 and b.z_at(1, 255) == -0.25, "entry 1's z is its own")

    # ── a MISS must not be a near miss ────────────────────────────────
    # A voice layer guessing at the nearest command is how a robot ends up
    # doing something nobody asked for.
    t.truth(b.find("squatt") == -1, "a near-miss name returns -1, not the nearest entry")
    t.truth(b.find("run") == -1, "a REJECTED command is absent, and absence reads as -1")
    t.truth(b.find("") == -1, "an empty name returns -1")

    # ── the width check, the whole point of the reader ────────────────
    write_text_atomic(path, _bank(256, 255, 256))
    var raised = False
    try:
        var bad = G1CommandBank.load(path)
        _ = bad.count()
    except:
        raised = True
    t.truth(raised, "a z row ONE VALUE SHORT raises instead of sliding every later command")

    write_text_atomic(path, _bank(256, 256, 257))
    raised = False
    try:
        var bad2 = G1CommandBank.load(path)
        _ = bad2.count()
    except:
        raised = True
    t.truth(raised, "a z row one value LONG raises too")

    # ── a header-less file must not read as an empty bank ─────────────
    write_text_atomic(path, String("# no count line\nname squat\ngroup posture\n"))
    raised = False
    try:
        var bad3 = G1CommandBank.load(path)
        _ = bad3.count()
    except:
        raised = True
    t.truth(raised, "a file with no `count` header raises rather than loading empty")

    remove_file(path)
    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_g1_command_bank: " + String(t.fails) + " failed")
