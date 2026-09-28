"""PER-TASK SUCCESS RATES — the number an eval actually reports. P4b.

    var rep = SuccessReport(tbl)
    for lane in range(N): rep.record(lane, solved[lane])
    rep.show()
    rep.log_to(logger, step)
    if not rep.same_as(previous): raise ...

`TASK_LAYER_PLAN.md` §6.2 and P4's gate: *two runs of one policy on one frozen
init table report the SAME success rate, and the monitor breaks it down per
task.* This is the bookkeeping half of that sentence — deliberately separated
from the driver, because the driver needs a GPU and this does not.

## ⚠ AN AGGREGATE RATE OVER A MULTI-TASK TABLE IS ALMOST MEANINGLESS

A family's tasks are not equally hard, so one number over all of them moves
when the MIX moves, not only when the policy does. Freeze eight `reach`
episodes beside eight `lift` and the aggregate says more about the ratio than
about the arm. The per-task breakdown is the reportable quantity; the
aggregate is printed beside it because people ask for it, and it carries its
episode count so the mix is visible.

## ⚠⚠ COMPARING TWO RUNS COMPARES THE PER-LANE VECTOR, NOT THE RATE

`same_as` exists because "the same success rate" is the WEAK form of the claim
this gates. Two runs can score 12/16 on entirely different lanes — a policy
that is genuinely nondeterministic passes a rate comparison roughly whenever
the counts happen to land. So the default comparison is lane by lane, and the
rate is derived from it.

⚠ THE ORDER OF LANES IS THE ORDER OF THE TABLE'S ROWS, and that is what makes
lane-wise comparison meaningful across runs: lane `i` is init row `i` in both.
A driver that shuffled rows between runs would break this without breaking any
rate, so `record` takes the ROW INDEX and not "the next lane".
"""

from noeira.core.logger import Logger
from .init_table import InitTable


struct SuccessReport(Movable & Deinitable):
    """Per-lane outcomes over one init table, and the per-task rates."""

    var solved: List[Bool]
    var seen: List[Bool]
    """Whether `record` was called for each lane.

    ⚠ NOT A CONVENIENCE. An eval that skipped lanes — an early `break`, a
    driver whose batch is smaller than the table — would otherwise report
    those lanes as FAILURES and print a plausible, lower success rate.
    `show`/`log_to`/`rate` refuse an incomplete report instead."""
    var task_index: List[Int32]
    var labels: List[String]

    def __init__(out self, tbl: InitTable) raises:
        self.solved = List[Bool]()
        self.seen = List[Bool]()
        self.task_index = List[Int32]()
        self.labels = List[String]()
        for i in range(tbl.n_rows()):
            self.solved.append(False)
            self.seen.append(False)
            self.task_index.append(tbl.task_index[i])
        # ⚠ RESOLVED ONCE, AT CONSTRUCTION, so an unnamed `task_index` fails
        # HERE — before the eval runs — rather than after it, when the run has
        # already been paid for and the only thing missing is the label.
        var hi = -1
        for i in range(tbl.n_rows()):
            if Int(tbl.task_index[i]) > hi:
                hi = Int(tbl.task_index[i])
        for _ in range(hi + 1):
            self.labels.append(String(""))
        for i in range(tbl.n_rows()):
            self.labels[Int(tbl.task_index[i])] = tbl.task_label(i)

    def __init__(out self, *, deinit move: Self):
        self.solved = move.solved^
        self.seen = move.seen^
        self.task_index = move.task_index^
        self.labels = move.labels^

    def n_lanes(self) -> Int:
        return len(self.solved)

    def record(mut self, lane: Int, solved: Bool) raises:
        """Lane `lane` — which is init row `lane` — solved its goal or did not.

        ⚠ RECORDING A LANE TWICE RAISES. A driver that called this inside its
        step loop instead of after it would otherwise report the LAST step's
        answer, and for a task that terminates on success the last step is the
        one after the episode ended.
        """
        if lane < 0 or lane >= self.n_lanes():
            raise Error(
                "tasks: eval lane " + String(lane) + " out of range (table has "
                + String(self.n_lanes()) + " rows)"
            )
        if self.seen[lane]:
            raise Error(
                "tasks: eval lane " + String(lane) + " recorded twice. Record"
                " each lane ONCE, after its episode — a per-step call reports"
                " the final step, not whether the goal was ever met."
            )
        self.seen[lane] = True
        self.solved[lane] = solved

    def _require_complete(self) raises:
        var missing = 0
        for i in range(self.n_lanes()):
            if not self.seen[i]:
                missing += 1
        if missing != 0:
            raise Error(
                "tasks: " + String(missing) + " of " + String(self.n_lanes())
                + " eval lanes were never recorded. An unrecorded lane is not"
                " a failed one — reporting it as such would print a lower rate"
                " that looks like a worse policy."
            )

    def n_tasks(self) -> Int:
        return len(self.labels)

    def counts(self, task: Int) raises -> Tuple[Int, Int]:
        """`(solved, total)` for one `task_index`."""
        self._require_complete()
        var s = 0
        var n = 0
        for i in range(self.n_lanes()):
            if Int(self.task_index[i]) == task:
                n += 1
                if self.solved[i]:
                    s += 1
        return (s, n)

    def rate(self, task: Int) raises -> Float64:
        var c = self.counts(task)
        if c[1] == 0:
            raise Error(
                "tasks: no eval lanes for task_index " + String(task)
                + " — a rate over zero episodes prints as 0.0 and reads as a"
                " failing policy."
            )
        return Float64(c[0]) / Float64(c[1])

    def overall(self) raises -> Float64:
        self._require_complete()
        var s = 0
        for i in range(self.n_lanes()):
            if self.solved[i]:
                s += 1
        return Float64(s) / Float64(self.n_lanes())

    def label(self, task: Int) raises -> String:
        if task < 0 or task >= len(self.labels) or self.labels[task] == "":
            raise Error(
                "tasks: task_index " + String(task) + " has no instruction."
            )
        return String(self.labels[task])

    def same_as(self, other: Self) raises -> Bool:
        """Lane by lane, not rate by rate. See the module header."""
        self._require_complete()
        other._require_complete()
        if self.n_lanes() != other.n_lanes():
            return False
        for i in range(self.n_lanes()):
            if self.solved[i] != other.solved[i]:
                return False
            if self.task_index[i] != other.task_index[i]:
                return False
        return True

    def n_differing_lanes(self, other: Self) raises -> Int:
        """How many lanes disagree — what a diagnostic prints when `same_as`
        is False, and what a CONTROL asserts is nonzero."""
        self._require_complete()
        other._require_complete()
        var n = self.n_lanes()
        if other.n_lanes() < n:
            n = other.n_lanes()
        var d = 0
        for i in range(n):
            if self.solved[i] != other.solved[i]:
                d += 1
        return d + (self.n_lanes() - n)

    def show(self, title: String = String("success")) raises:
        self._require_complete()
        print("  " + title + " — " + String(self.n_lanes()) + " episodes")
        for t in range(self.n_tasks()):
            if self.labels[t] == "":
                continue
            var c = self.counts(t)
            if c[1] == 0:
                continue
            print(
                "    task", t, ":", c[0], "/", c[1], "=",
                Float64(c[0]) / Float64(c[1]), " | " + self.labels[t]
            )
        print("    overall :", self.overall())

    def log_to[L: Logger](self, mut logger: L, step: Int) raises:
        """Per-task rates to the monitor, plus the aggregate.

        ⚠ THE METRIC NAME CARRIES THE TASK'S INDEX, NOT ITS INSTRUCTION. An
        instruction is free text — it has spaces, punctuation and, in this
        tree's own datasets, non-ASCII — and a metric key that changes when
        somebody rewords a `language=` line breaks every chart built on it.
        The index is stable for the life of an init table, which is exactly
        the period over which the numbers are meant to be comparable.

        ⚠ THE EPISODE COUNT IS LOGGED BESIDE EACH RATE. A rate with no
        denominator cannot be told from the same rate over a third as many
        episodes, and a per-task breakdown is precisely where the denominators
        differ.
        """
        self._require_complete()
        for t in range(self.n_tasks()):
            if self.labels[t] == "":
                continue
            var c = self.counts(t)
            if c[1] == 0:
                continue
            var k = String("eval_success_rate_task_") + String(t)
            logger.log_scalar(k, Float64(c[0]) / Float64(c[1]), step)
            logger.log_scalar(
                String("eval_episodes_task_") + String(t), Float64(c[1]), step
            )
        logger.log_scalar(
            String("eval_success_rate"), self.overall(), step
        )
        logger.log_scalar(
            String("eval_episodes"), Float64(self.n_lanes()), step
        )
