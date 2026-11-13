# +--------------------------------------------------------------------------+ #
# | Tasks and the executive — steps run in order, by plain code
# +--------------------------------------------------------------------------+ #
"""What the robot has been asked to do, as steps, and the code that runs them
(docs/ROBOT_BRAIN_PLAN.md §5.6, step B2).

    var task = Task(String("va à la table puis à la porte"))
    task.steps.append(Step.goto("table"))
    task.steps.append(Step.goto("door"))
    give_task(ws, task^, t, now)                 # replaces the task in hand

    # once a frame, after the frame's facts were observed (`ws.observe_motion`):
    var act = tick_tasks(ws, t, now)
    if act.kind == EX_START:    body.apply(act.command)       # "goto:table"
    elif act.kind == EX_NOTICE:
        if act.command != "":   body.apply(act.command)       # "stop": the walk that failed ends
        voice.say(notice_text(ws, act))                        # "Je n'ai pas réussi à atteindre la table."

⚠ NO MODEL HERE. A model answers a typed question upstream (which place,
which command, in which order); from there on the steps, their order, their
deadlines and what is said on a failure are code. A 2026 study of a
planner-executor loop on robots found it fragile on long tasks with a far
larger model than ours (plan §4), and our own models are measured on
constrained answers.

A STEP is one command of the body, as the body takes it (`goto:table`,
`squat`), with how its end is known:
- a step to a PLACE is done when the state's `at` is that place, and has
  failed when `timeout_s` passed first;
- a step with no place is done after `hold_s`.
A command the body refuses fails its step (`refuse_step`).

THE CLOCK is the caller's, in seconds (`t`): simulated time in a headless run,
so a scripted episode replays exactly. `now` (epoch seconds) only dates the
events. The task in hand and the last one finished live in the `WorldState`
(`ws.task`, `ws.last_task`); the reply view reads them, so "what are you
doing?" and "did you get there?" are answered from the same record.

⚠ THE LAST REQUEST WINS: `give_task` cancels the task in hand, as a new
spoken command always replaced the running one.
"""

from .events import EV_TASK_STARTED, EV_STEP_STARTED, EV_STEP_DONE, EV_TASK_DONE, EV_TASK_FAILED, EV_TASK_CANCELLED
from .state import WorldState, Task, Step, TASK_RUNNING, TASK_DONE, TASK_FAILED, TASK_CANCELLED, STEP_PENDING, STEP_RUNNING, STEP_DONE, STEP_FAILED


comptime EX_NONE: Int = 0
comptime EX_START: Int = 1
"""Give the body `command` now."""
comptime EX_NOTICE: Int = 2
"""Something to tell the person: `notice` says what, `place` about where.
`command` is `STOP_COMMAND` when the body must also be stopped."""

comptime STOP_COMMAND = "stop"
"""What the body is given when a step fails while it is moving. ⚠ Without it
the robot said "I could not reach the door" and went on walking to it (the
first scripted episode on the Orin, docs/ROBOT_BRAIN_PLAN.md B2)."""

comptime NOTICE_NOT_REACHED = "not_reached"
"""A step's place was not reached in time."""
comptime NOTICE_REFUSED = "refused"
"""The body does not know a step's command."""
comptime NOTICE_ALL_DONE = "all_done"
"""A task of several steps is finished (a single step ends silently: its
arrival is its own report)."""


@fieldwise_init
struct ExecAction(Copyable, Movable):
    var kind: Int
    var command: String
    var notice: String
    var place: String

    @staticmethod
    def none() -> ExecAction:
        return ExecAction(EX_NONE, String(""), String(""), String(""))


def _finish(mut ws: WorldState, status: Int, reason: String, t: Float64, now: Int):
    """The task in hand becomes the last task."""
    ws.task.status = status
    ws.task.reason = reason
    ws.task.t_end = t
    var what = ws.task.describe()
    if status == TASK_DONE:
        ws.record(EV_TASK_DONE, what, String("executive"), now)
    elif status == TASK_FAILED:
        ws.record(EV_TASK_FAILED, what + " | " + reason, String("executive"), now)
    else:
        ws.record(EV_TASK_CANCELLED, what + " | " + reason, String("executive"), now)
    ws.last_task = ws.task.copy()
    ws.has_last_task = True
    ws.has_task = False


def give_task(mut ws: WorldState, var task: Task, t: Float64, now: Int):
    """`task` becomes the task in hand; the one it replaces is cancelled. A
    task with no step is not taken."""
    if len(task.steps) == 0:
        return
    if ws.has_task:
        _finish(ws, TASK_CANCELLED, String("a new request replaced it"), t, now)
    task.status = TASK_RUNNING
    task.at = 0
    task.t_start = t
    for i in range(len(task.steps)):
        task.steps[i].status = STEP_PENDING
    ws.task = task^
    ws.has_task = True
    var what = ws.task.describe()
    ws.record(EV_TASK_STARTED, what, String("executive"), now)


def cancel_task(mut ws: WorldState, reason: String, t: Float64, now: Int):
    """Drop the task in hand ("stop", or a command given outside the tasks)."""
    if ws.has_task:
        _finish(ws, TASK_CANCELLED, reason, t, now)


def refuse_step(mut ws: WorldState, t: Float64, now: Int) -> ExecAction:
    """The body did not take the command `tick_tasks` just started: the step
    and its task fail, and the person is told."""
    if not ws.has_task:
        return ExecAction.none()
    var k = ws.task.at
    ws.task.steps[k].status = STEP_FAILED
    var cmd = ws.task.steps[k].command.copy()
    var place = ws.task.steps[k].place.copy()
    _finish(ws, TASK_FAILED, "the robot has no command '" + cmd + "'", t, now)
    return ExecAction(EX_NOTICE, String(""), String(NOTICE_REFUSED), place)


def tick_tasks(mut ws: WorldState, t: Float64, now: Int) -> ExecAction:
    """One frame of the task in hand: at most one thing to do comes back."""
    if not ws.has_task:
        return ExecAction.none()
    var k = ws.task.at
    if ws.task.steps[k].status == STEP_PENDING:
        ws.task.steps[k].status = STEP_RUNNING
        ws.task.steps[k].t_start = t
        var started = ws.task.steps[k].label.copy()
        ws.record(EV_STEP_STARTED, started, String("executive"), now)
        return ExecAction(EX_START, ws.task.steps[k].command.copy(), String(""), ws.task.steps[k].place.copy())
    var place = ws.task.steps[k].place.copy()
    var elapsed = t - ws.task.steps[k].t_start
    var done = False
    if place != "":
        if ws.at == place:
            done = True
        elif elapsed > ws.task.steps[k].timeout_s:
            ws.task.steps[k].status = STEP_FAILED
            _finish(
                ws, TASK_FAILED,
                "the " + place + " was not reached after " + String(Int(ws.task.steps[k].timeout_s)) + " seconds", t, now,
            )
            return ExecAction(EX_NOTICE, String(STOP_COMMAND), String(NOTICE_NOT_REACHED), place)
    elif elapsed >= ws.task.steps[k].hold_s:
        done = True
    if not done:
        return ExecAction.none()
    ws.task.steps[k].status = STEP_DONE
    var finished = ws.task.steps[k].label.copy()
    ws.record(EV_STEP_DONE, finished, String("executive"), now)
    if k + 1 < len(ws.task.steps):
        ws.task.at = k + 1
        return ExecAction.none()  # the next step starts on the next frame
    var several = len(ws.task.steps) > 1
    _finish(ws, TASK_DONE, String(""), t, now)
    if several:
        return ExecAction(EX_NOTICE, String(""), String(NOTICE_ALL_DONE), place)
    return ExecAction.none()


def notice_text(ref ws: WorldState, act: ExecAction) -> String:
    """What the robot says for a notice, in the conversation's language
    (French and English; English otherwise). ⚠ A TEMPLATE, not a model's
    sentence: it reports a fact the executive holds exactly, at once."""
    var fr = ws.language == "fr"
    var place = ws.word_for(act.place, ws.language)
    if act.notice == NOTICE_NOT_REACHED:
        if fr:
            return "Je n'ai pas réussi à atteindre " + place + "."
        return "I could not reach " + place + "."
    if act.notice == NOTICE_REFUSED:
        return String("Je ne sais pas faire ça.") if fr else String("I do not know how to do that.")
    if act.notice == NOTICE_ALL_DONE:
        return String("C'est fait.") if fr else String("Done.")
    return String("")
