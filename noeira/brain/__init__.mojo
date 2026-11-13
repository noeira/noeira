"""The robot's brain (docs/ROBOT_BRAIN_PLAN.md): one shared state, its record,
and the text each model reads from it.

    from noeira.brain import WorldState, Body, PlaceXY, reply_view, robot_dir

B1: `WorldState` (what the robot is, where it is, what it is doing, what
happened), `EventLog` (the append-only record on disk), `reply_view` (the reply
model's prompt, written from the state) and the place geometry the decision
model's lines are made of. B2: `Task` / `Step` and the executive that runs them
(`give_task`, `tick_tasks`: steps in order, arrivals, deadlines, what to say on
a failure), with the task in hand in the reply view. Persistence across
restarts (B3), people (B4) and a second body (B5) are not here yet.
"""

from .geometry import (
    PlaceXY, PlaceAround, places_around, place_line, place_lines, nearest_place,
    bearing_word, wrap_deg, NEAR_M, NEXT_TO_M,
)
from .events import (
    BrainEvent, EventLog, event_json, parse_event, load_events, robot_dir,
    EV_SESSION, EV_HEARD, EV_SAID, EV_BEGAN, EV_GOING, EV_ARRIVED,
    EV_TASK_STARTED, EV_STEP_STARTED, EV_STEP_DONE, EV_TASK_DONE, EV_TASK_FAILED, EV_TASK_CANCELLED,
)
from .state import (
    WorldState, Body, PlaceWord, Task, Step, WS_EVENTS_N, WS_TRAIL_N,
    TASK_RUNNING, TASK_DONE, TASK_FAILED, TASK_CANCELLED, STEP_PENDING, STEP_RUNNING, STEP_DONE, STEP_FAILED,
)
from .tasks import (
    ExecAction, give_task, cancel_task, refuse_step, tick_tasks, notice_text,
    EX_NONE, EX_START, EX_NOTICE, STOP_COMMAND, NOTICE_NOT_REACHED, NOTICE_REFUSED, NOTICE_ALL_DONE,
)
from .views import ReplyView, reply_view, reply_stable, reply_live, task_sentences, metres, language_name, REPLY_VIEW_LINES, REPLY_VIEW_SENTENCES
