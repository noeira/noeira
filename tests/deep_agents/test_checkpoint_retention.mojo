"""The retention rule decides what gets DELETED, so it gets a gate.

    pixi run mojo run -I . tests/deep_agents/test_checkpoint_retention.mojo

WHY THIS EXISTS
===============
The 2048/6 G1 run died of a full disk at 38 000 batched steps: 19 checkpoints
of 1.17 GB each on a 60 GB box, and the 19th cut off mid-write. Twenty hours
and $16 of rented 5090, with nothing wrong upstream of the filesystem.

`checkpoint_retain` is the fix, and it is the one piece of that fix that can
be silently wrong in the dangerous direction. A rule that keeps too much only
wastes space; a rule that keeps too LITTLE deletes the checkpoint you were
going to resume from, and you find out when the box dies.

⚠ THE G1 TRAINER CANNOT BE BUILT ON THE LAPTOP — its batched env is NVIDIA
only and Metal rejects the float64 in its kernels — so the wiring there is not
coverable here. Extracting the policy into `deep_agents/training/checkpoint`
is what makes this arm testable at all, and it is why the rule lives in a
library rather than beside its single caller.
"""

from noeira.deep_agents.training.checkpoint import checkpoint_retain


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


def _steps(first: Int, last: Int, every: Int) -> List[Int]:
    var v = List[Int]()
    var s = first
    while s <= last:
        v.append(s)
        s += every
    return v^


def main() raises:
    var t = Tally()
    print("=== checkpoint retention ===")

    # the run that died: every 2000 steps out to 38 000
    var w = _steps(2000, 38000, 2000)
    t.truth(len(w) == 19, String("the fixture is the run that died (") + String(len(w)) + String(" checkpoints)"))

    # ── the defaults the trainer ships ────────────────────────────────
    var keep = 3
    var mile = 25000
    var best = 33792  # not on any ladder, and not recent — the hard case
    var alive = List[Int]()
    for i in range(len(w)):
        if checkpoint_retain(w[i], w, keep, mile, best):
            alive.append(w[i])
    print("     survivors:", len(alive), "of", len(w))
    t.truth(
        len(alive) < len(w),
        String("the policy actually prunes (") + String(len(alive))
        + String(" of ") + String(len(w)) + String(")"),
    )

    # ⚠ THE THREE REASONS, EACH CHECKED SEPARATELY. A policy that kept
    # everything would pass a "the newest survives" test on its own.
    t.truth(checkpoint_retain(38000, w, keep, mile, best), "the NEWEST survives (resume)")
    t.truth(checkpoint_retain(36000, w, keep, mile, best), "second newest survives")
    t.truth(checkpoint_retain(34000, w, keep, mile, best), "third newest survives (keep=3)")
    t.truth(not checkpoint_retain(32000, w, keep, mile, best), "the FOURTH newest does not")
    t.truth(checkpoint_retain(25000, w, keep, mile, best), "a milestone survives (25000 % 25000)")
    t.truth(not checkpoint_retain(26000, w, keep, mile, best), "a non-milestone mid-run does not")

    # best is the one the run actually loses without a rule for it: §12.44's
    # best score was at a step whose checkpoint was NOT kept.
    # ⚠ INSERTED IN ORDER, not appended. `written` is oldest-first and "the
    # last keep" is POSITIONAL, so appending 33792 to the end would make it
    # the most recent entry and `keep=3` would protect it whatever `best`
    # said — the arm would pass while testing nothing. It belongs between
    # 32000 and 34000, where a mid-run best actually sits.
    var w2 = List[Int]()
    for i in range(len(w)):
        if w[i] == 34000:
            w2.append(33792)
        w2.append(w[i])
    t.truth(
        checkpoint_retain(33792, w2, keep, mile, 33792),
        "the BEST survives even off the ladder and out of the recent window",
    )
    t.truth(
        not checkpoint_retain(33792, w2, keep, mile, -1),
        "...and only because it is best — with best=-1 it is dropped",
    )

    # ── boundary conditions that would delete the wrong thing ─────────
    t.truth(
        checkpoint_retain(38000, w, 1, 0, -1),
        "keep=1, no ladder, no best: the newest still survives",
    )
    t.truth(
        not checkpoint_retain(36000, w, 1, 0, -1),
        "keep=1: the second newest does not",
    )
    var everything = True
    for i in range(len(w)):
        if not checkpoint_retain(w[i], w, len(w), 0, -1):
            everything = False
    t.truth(everything, "keep >= n keeps everything (no off-by-one at the head)")

    var nothing = 0
    for i in range(len(w)):
        if checkpoint_retain(w[i], w, 0, 0, -1):
            nothing += 1
    t.truth(nothing == 0, String("keep=0, no ladder, no best keeps nothing (") + String(nothing) + String(")"))

    # ⚠ A SHORT `written` MUST NOT UNDERFLOW. After a prune the list shrinks,
    # and the next call passes the shrunken list — if "last keep" were
    # computed as an index rather than a clamped slice this is where it would
    # read out of range or keep nothing.
    var short = List[Int]()
    short.append(38000)
    t.truth(
        checkpoint_retain(38000, short, 3, 0, -1),
        "a 1-element list with keep=3 keeps its only entry",
    )
    var empty = List[Int]()
    t.truth(
        not checkpoint_retain(38000, empty, 3, 0, -1),
        "an empty list keeps nothing and does not crash",
    )

    # ── the property that matters: a long run stays bounded ───────────
    # 64 h at 2000-step cadence is ~192 checkpoints ~ 225 GB unpruned.
    var long_w = _steps(2000, 384000, 2000)
    var survivors = 0
    for i in range(len(long_w)):
        if checkpoint_retain(long_w[i], long_w, 3, 25000, 200000):
            survivors += 1
    print("     64 h run:", len(long_w), "written ->", survivors, "kept ~",
          Float64(survivors) * 1.17, "GB")
    t.truth(
        survivors <= 24,
        String("a 64 h run stays bounded (") + String(survivors)
        + String(" kept, ~") + String(Float64(survivors) * 1.17)
        + String(" GB, against ") + String(len(long_w)) + String(")"),
    )

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_checkpoint_retention: " + String(t.fails) + " failed")
