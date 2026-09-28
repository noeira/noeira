"""PER-TASK SUCCESS RATES, and what reaches the monitor. P4b (host half).

P4's gate is *two runs of one policy on one frozen init table report the SAME
success rate, and the monitor breaks it down per task.* The DRIVER needs a GPU
(`examples/tasks/task_eval_frozen.mojo`); the bookkeeping does not, and this
is the bookkeeping.

## WHAT IT ASSERTS

1. **THE BREAKDOWN IS ARITHMETIC, PER TASK**, against counts computed here
   from the same outcome vector by a different route.

2. ⚠⚠ **`same_as` COMPARES LANES, NOT RATES.** Two runs scoring 12/16 on
   DIFFERENT lanes is a nondeterministic policy passing a rate comparison by
   coincidence. The gate builds exactly that pair — equal rates, disjoint
   lanes — and demands `same_as` say False. A rate-based comparison passes it
   and is the reason this check exists.

3. **AN INCOMPLETE REPORT RAISES.** A lane that was never recorded is not a
   failed lane; counting it as one prints a lower rate that reads as a worse
   policy. Same for a lane recorded twice.

4. **WHAT REACHES THE MONITOR IS READ BACK OFF DISK**, through `CsvLogger` —
   not asserted from the call site. The metric keys, the per-task rates and
   the EPISODE COUNTS beside them are checked as rows in the file.

5. ⚠ **ANTI-VACUITY THROUGHOUT.** The two tasks are given DIFFERENT rates, so
   a breakdown that reported one number twice fails; and the counts differ
   from each other, so a denominator dropped from the log is visible.

Run: pixi run mojo run -I . tests/tasks/test_eval_report.mojo
"""

from std.pathlib import Path

from noeira.core.logger import CsvLogger
from noeira.core.kv import split_on
from noeira.tasks.init_table import (
    InitTable, write_init_table, load_init_table,
)
from noeira.tasks.eval_report import SuccessReport


comptime NQ = 4
comptime NV = 3
comptime FAMILY = "unit_family"
comptime OUT = "/tmp/noeira_eval_report.h5"
comptime CSV = "/tmp/noeira_eval_report.csv"

# ⚠ DELIBERATELY UNEQUAL. Task 0 gets 6 episodes and task 1 gets 4, so a log
# that dropped the denominators, or a breakdown that reused one task's count
# for the other, is visible rather than arithmetically invisible.
comptime N0 = 6
comptime N1 = 4
comptime N = N0 + N1


struct Tally(Copyable, ImplicitlyCopyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def check(mut self, ok: Bool, what: String):
        self.checks += 1
        if ok:
            print("  ok:", what)
        else:
            self.failures += 1
            print("  FAIL:", what)


def _table() raises -> InitTable:
    """A tiny synthetic init table — this file gates the REPORT, not the
    sampler, so the states are arbitrary and only the task column matters."""
    var words = 1 + NQ + NV
    var st = List[Float64]()
    var tix = List[Int32]()
    var mk = List[Float64]()
    for i in range(N):
        for w in range(words):
            st.append(Float64(i * 100 + w))
        tix.append(Int32(0) if i < N0 else Int32(1))
        mk.append(3.0)
    var names = List[String]()
    names.append(String("Put both blocks on the table"))
    names.append(String("Move the gripper over the table"))
    write_init_table(String(OUT), String(FAMILY), NQ, NV, st, tix, mk, names)
    return load_init_table(String(OUT), String(FAMILY), NQ, NV)


def _csv_value(text: String, key: String) raises -> Float64:
    """The `value` of the last row whose `name` is `key`. Raises if absent."""
    var lines = split_on(text, String("\n"))
    var found = False
    var out = 0.0
    for i in range(len(lines)):
        var row = String(lines[i].strip())
        if row.byte_length() == 0:
            continue
        var c = split_on(row, String(","))
        if len(c) != 4:
            continue
        if String(c[2]) == key:
            out = Float64(String(c[3]))
            found = True
    if not found:
        raise Error("no metric named '" + key + "' in the log")
    return out


def main() raises:
    print("=== per-task success rates, and what reaches the monitor ===")
    var ta = Tally()

    var tbl = _table()
    print()
    print("--- 1. the breakdown ---")
    print("    table:", tbl.n_rows(), "rows —", N0, "of task 0,", N1,
          "of task 1")

    # Task 0: solve lanes 0,1,2  -> 3/6.  Task 1: solve lane 6 -> 1/4.
    var rep = SuccessReport(tbl)
    for i in range(N):
        rep.record(i, (i < 3) or (i == N0))
    rep.show(String("run A"))

    var c0 = rep.counts(0)
    var c1 = rep.counts(1)
    ta.check(c0[0] == 3 and c0[1] == N0,
             "task 0: " + String(c0[0]) + " / " + String(c0[1]))
    ta.check(c1[0] == 1 and c1[1] == N1,
             "task 1: " + String(c1[0]) + " / " + String(c1[1]))
    ta.check(rep.rate(0) == 0.5 and rep.rate(1) == 0.25,
             "rates 0.5 and 0.25")
    # ⚠ ANTI-VACUITY: equal rates would let a breakdown that computes one
    # number and prints it twice pass every line above.
    ta.check(rep.rate(0) != rep.rate(1),
             "the two tasks have DIFFERENT rates, so the split is real")
    ta.check(rep.overall() == 4.0 / Float64(N),
             "overall " + String(rep.overall()) + " = 4/" + String(N))
    ta.check(rep.label(0) != rep.label(1),
             "each task's instruction came through: '" + rep.label(0) + "'")

    # ── 2. same_as compares LANES, not rates ─────────────────────────────
    print()
    print("--- 2. two runs with the SAME rate on DIFFERENT lanes ---")
    var same = SuccessReport(tbl)
    for i in range(N):
        same.record(i, (i < 3) or (i == N0))
    ta.check(rep.same_as(same), "an identical outcome vector compares EQUAL")

    # Task 0: solve lanes 3,4,5 instead — still 3/6. Task 1: lane 7 — still 1/4.
    var shifted = SuccessReport(tbl)
    for i in range(N):
        shifted.record(i, (i >= 3 and i < N0) or (i == N0 + 1))
    ta.check(shifted.rate(0) == rep.rate(0) and shifted.rate(1) == rep.rate(1),
             "the shifted run has the IDENTICAL per-task rates")
    # ⚠⚠ THE CHECK THIS FILE EXISTS FOR. A rate comparison passes here.
    ta.check(not rep.same_as(shifted),
             "and `same_as` still says NO — it compares lanes, not rates")
    ta.check(rep.n_differing_lanes(shifted) == 8,
             String(rep.n_differing_lanes(shifted))
             + " lanes differ between them")

    # ── 3. an incomplete or double-counted report raises ─────────────────
    print()
    print("--- 3. a lane never recorded is not a lane that failed ---")
    var partial = SuccessReport(tbl)
    for i in range(N - 1):
        partial.record(i, True)
    var raised = 0
    try:
        var _r = partial.overall()
        print("    NOT RAISED: overall() on a report missing a lane")
    except e:
        raised += 1
        print("    raised on overall() with 1 lane unrecorded")
    try:
        partial.show(String("partial"))
        print("    NOT RAISED: show() on an incomplete report")
    except e:
        raised += 1
        print("    raised on show()")
    var dup = SuccessReport(tbl)
    dup.record(0, True)
    try:
        dup.record(0, False)
        print("    NOT RAISED: the same lane recorded twice")
    except e:
        raised += 1
        print("    raised on a double record()")
    ta.check(raised == 3, String(raised) + " of 3 misuses raised")

    # ── 4. what actually reaches the monitor ─────────────────────────────
    print()
    print("--- 4. read the metrics back off disk ---")
    # ⚠ A STALE LOG WOULD BE APPENDED TO, and `_csv_value` reads the LAST
    # row with a given name — so a previous run's numbers would be shadowed
    # rather than read, and only a CHANGE in the expected values would show
    # it. Truncate by writing an empty file rather than trusting the logger.
    var p = Path(String(CSV))
    with open(String(CSV), "w") as _f:
        _f.write(String(""))
    var logger = CsvLogger(String(CSV))
    rep.log_to(logger, 42)
    logger.close()

    var text = p.read_text()
    var n_rows = 0
    var lines = split_on(text, String("\n"))
    for i in range(len(lines)):
        if String(lines[i].strip()).byte_length() > 0:
            n_rows += 1
    print("    ", n_rows - 1, "metric rows written")

    var r0 = _csv_value(text, String("eval_success_rate_task_0"))
    var r1 = _csv_value(text, String("eval_success_rate_task_1"))
    var e0 = _csv_value(text, String("eval_episodes_task_0"))
    var e1 = _csv_value(text, String("eval_episodes_task_1"))
    var ov = _csv_value(text, String("eval_success_rate"))
    var oe = _csv_value(text, String("eval_episodes"))
    print("     task_0:", r0, "over", e0, " task_1:", r1, "over", e1,
          " overall:", ov, "over", oe)
    ta.check(r0 == 0.5 and r1 == 0.25,
             "the per-task rates reached the log")
    # ⚠ THE DENOMINATOR IS THE HALF THAT GETS DROPPED. A rate with no episode
    # count cannot be told from the same rate over a third as many episodes,
    # and a breakdown is exactly where the counts differ.
    ta.check(e0 == Float64(N0) and e1 == Float64(N1),
             "and so did the EPISODE COUNTS, " + String(e0) + " and "
             + String(e1))
    ta.check(e0 != e1, "which differ, so a dropped denominator is visible")
    ta.check(ov == 4.0 / Float64(N) and oe == Float64(N),
             "the aggregate reached it too, with its own count")

    var missing = 0
    try:
        var _v = _csv_value(text, String("eval_success_rate_task_2"))
        print("    NOT RAISED: a task the table never had")
    except e:
        missing += 1
    ta.check(missing == 1,
             "and NOTHING was logged for a task the table does not have")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "eval report: " + String(ta.failures) + " of " + String(ta.checks)
            + " failed"
        )
    print("=== PASS ===")
