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
- a step about an OBJECT (`Step.pick`, `Step.put`) is done on the object's
  record: held for a pick, put down at the step's place for a put
  (`noeira.brain.objects`); failed after `timeout_s`, and at once, before the
  body is given anything, when the state has no pose for the object or it is
  missing. A put step whose object was held and then let go anywhere but at
  its place has failed at once (`NOTICE_PUT_ELSEWHERE`): no timeout to wait.
A command the body refuses fails its step (`refuse_step`, with the body's
reason for the log); for an object step the notice names the object and the
place (« Je ne peux pas porter la caisse jusqu'à la table. »).

THE CLOCK is the caller's, in seconds (`t`): simulated time in a headless run,
so a scripted episode replays exactly. `now` (epoch seconds) only dates the
events. The task in hand and the last one finished live in the `WorldState`
(`ws.task`, `ws.last_task`); the reply view reads them, so "what are you
doing?" and "did you get there?" are answered from the same record.

⚠ THE LAST REQUEST WINS: `give_task` cancels the task in hand, as a new
spoken command always replaced the running one.
"""

from .events import EV_TASK_STARTED, EV_STEP_STARTED, EV_STEP_DONE, EV_TASK_DONE, EV_TASK_FAILED, EV_TASK_CANCELLED
from .objects import OBJ_HELD, OBJ_PUT
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
comptime NOTICE_NO_OBJECT = "no_object"
"""A step's object has no pose in the state, or it is missing: the robot does
not know where it is. Nothing was given to the body."""
comptime NOTICE_NOT_DONE = "not_done"
"""A step's object was not picked up (or not put at its place) in time."""
comptime NOTICE_PUT_ELSEWHERE = "put_elsewhere"
"""A put step's object was held, then let go somewhere other than its place."""


@fieldwise_init
struct ExecAction(Copyable, Movable):
    var kind: Int
    var command: String
    var notice: String
    var place: String
    var obj: String
    """The object the notice is about; "" for none."""

    @staticmethod
    def none() -> ExecAction:
        return ExecAction(EX_NONE, String(""), String(""), String(""), String(""))


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


def refuse_step(mut ws: WorldState, t: Float64, now: Int, why: String = "") -> ExecAction:
    """The body did not take the command `tick_tasks` just started: the step
    and its task fail, and the person is told. `why`: the body's reason, for
    the log ("" = it has no such command)."""
    if not ws.has_task:
        return ExecAction.none()
    var k = ws.task.at
    ws.task.steps[k].status = STEP_FAILED
    var cmd = ws.task.steps[k].command.copy()
    var place = ws.task.steps[k].place.copy()
    var obj = ws.task.steps[k].obj.copy()
    _finish(ws, TASK_FAILED, ("the robot has no command '" + cmd + "'") if why == "" else ("the body refused '" + cmd + "': " + why), t, now)
    return ExecAction(EX_NOTICE, String(""), String(NOTICE_REFUSED), place, obj)


def tick_tasks(mut ws: WorldState, t: Float64, now: Int) -> ExecAction:
    """One frame of the task in hand: at most one thing to do comes back."""
    if not ws.has_task:
        return ExecAction.none()
    var k = ws.task.at
    var obj = ws.task.steps[k].obj.copy()
    var oi = ws.object_index(obj) if obj != "" else -1
    if ws.task.steps[k].status == STEP_PENDING:
        if obj != "" and (oi < 0 or not ws.objects[oi].has_pose or ws.objects[oi].missing):
            ws.task.steps[k].status = STEP_FAILED
            var place0 = ws.task.steps[k].place.copy()
            _finish(ws, TASK_FAILED, "where the " + obj + " is is not known", t, now)
            return ExecAction(EX_NOTICE, String(""), String(NOTICE_NO_OBJECT), place0, obj)
        ws.task.steps[k].status = STEP_RUNNING
        ws.task.steps[k].t_start = t
        var started = ws.task.steps[k].label.copy()
        ws.record(EV_STEP_STARTED, started, String("executive"), now)
        return ExecAction(EX_START, ws.task.steps[k].command.copy(), String(""), ws.task.steps[k].place.copy(), obj)
    var place = ws.task.steps[k].place.copy()
    var elapsed = t - ws.task.steps[k].t_start
    var done = False
    if obj != "":
        # ⚠ BEFORE the place: a put step HAS a place, and the robot standing
        # there says nothing about where the object is
        if oi >= 0 and place == "":
            done = ws.objects[oi].where == OBJ_HELD
        elif oi >= 0:
            done = ws.objects[oi].where == OBJ_PUT and ws.objects[oi].put_at == place
            if ws.objects[oi].where == OBJ_HELD:
                ws.task.steps[k].held = True
            elif not done and ws.task.steps[k].held:
                # held, then let go elsewhere: the body has ended its carry
                ws.task.steps[k].status = STEP_FAILED
                var at = ws.objects[oi].put_at.copy()
                _finish(ws, TASK_FAILED, "the " + obj + " was put down " + (("at the " + at) if at != "" else String("at no place"))
                        + ", not at the " + place, t, now)
                return ExecAction(EX_NOTICE, String(""), String(NOTICE_PUT_ELSEWHERE), place, obj)
        if not done and elapsed > ws.task.steps[k].timeout_s:
            ws.task.steps[k].status = STEP_FAILED
            var what = ("the " + obj + " was not picked up") if place == "" else ("the " + obj + " was not put at the " + place)
            _finish(ws, TASK_FAILED, what + " after " + String(Int(ws.task.steps[k].timeout_s)) + " seconds", t, now)
            return ExecAction(EX_NOTICE, String(STOP_COMMAND), String(NOTICE_NOT_DONE), place, obj)
    elif place != "":
        if ws.at == place:
            done = True
        elif elapsed > ws.task.steps[k].timeout_s:
            ws.task.steps[k].status = STEP_FAILED
            _finish(
                ws, TASK_FAILED,
                "the " + place + " was not reached after " + String(Int(ws.task.steps[k].timeout_s)) + " seconds", t, now,
            )
            return ExecAction(EX_NOTICE, String(STOP_COMMAND), String(NOTICE_NOT_REACHED), place, String(""))
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
        return ExecAction(EX_NOTICE, String(""), String(NOTICE_ALL_DONE), place, String(""))
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
    if act.notice == NOTICE_ALL_DONE:
        return String("C'est fait.") if fr else String("Done.")
    var thing = ws.word_for(act.obj, ws.language)
    if act.notice == NOTICE_REFUSED:
        if act.obj == "":
            return String("Je ne sais pas faire ça.") if fr else String("I do not know how to do that.")
        if act.place == "":
            if fr:
                return "Je ne peux pas prendre " + thing + "."
            return "I cannot pick up " + thing + "."
        if fr:
            return "Je ne peux pas porter " + thing + " " + jusqua(place) + "."
        return "I cannot carry " + thing + " to " + place + "."
    if act.notice == NOTICE_PUT_ELSEWHERE:
        if fr:
            return "J'ai posé " + thing + ", mais pas " + pres_de(place) + "."
        return "I put " + thing + " down, but not at " + place + "."
    if act.notice == NOTICE_NO_OBJECT:
        if fr:
            return "Je ne sais pas où est " + thing + "."
        return "I do not know where " + thing + " is."
    if act.notice == NOTICE_NOT_DONE:
        if act.place == "":
            if fr:
                return "Je n'ai pas réussi à prendre " + thing + "."
            return "I could not pick up " + thing + "."
        if fr:
            return "Je n'ai pas réussi à porter " + thing + " " + jusqua(place) + "."
        return "I could not carry " + thing + " to " + place + "."
    return String("")


def pres_de(word: String) -> String:
    """« près de » before a French noun phrase: près de la table, près du frigo, près des chaises."""
    if word.startswith("le "):
        return "près du " + String(word[byte=3:])
    if word.startswith("les "):
        return "près des " + String(word[byte=4:])
    return "près de " + word


def jusqua(word: String) -> String:
    """« jusqu'à » before a French noun phrase, with its contractions:
    `la table` -> `jusqu'à la table`, `le frigo` -> `jusqu'au frigo`,
    `les chaises` -> `jusqu'aux chaises`."""
    if word.startswith("le "):
        return "jusqu'au " + String(word[byte=3:])
    if word.startswith("les "):
        return "jusqu'aux " + String(word[byte=4:])
    return "jusqu'à " + word
