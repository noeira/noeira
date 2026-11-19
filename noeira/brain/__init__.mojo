"""The robot's brain (docs/ROBOT_BRAIN_PLAN.md): one shared state, its record,
and the text each model reads from it.

    from noeira.brain import WorldState, Body, PlaceXY, reply_view, robot_dir

B1: `WorldState` (what the robot is, where it is, what it is doing, what
happened), `EventLog` (the append-only record on disk), `reply_view` (the reply
model's prompt, written from the state) and the place geometry the decision
model's lines are made of. B2: `Task` / `Step` and the executive that runs them
(`give_task`, `tick_tasks`: steps in order, arrivals, deadlines, what to say on
a failure), with the task in hand in the reply view. B3: `restore_memory` (what
the robot did on earlier days and sessions, rebuilt from its log by rule at
start) and `save_places` / `load_places`. B4: `People` (who gave their name,
whose voice this is, on disk) and the reply view's line about the person
talking. `computed_turn`: the questions code answers exactly (who the
speaker is, whether the task is finished, what the body can do — `Skill`,
`Body.add_motion` — and whether it can do one thing, run when it can), and
`reply_sentences`: how long the model's own reply may be. Movable objects
(`ObjectXY`, `noeira.brain.objects`): one record each, written from sightings,
the hands and what is told, the `pick:` / `put:` steps that end on them, and
`restore_objects`. A second body (B5) is not here yet.
"""

from .geometry import (
    PlaceXY, PlaceAround, places_around, place_line, place_lines, nearest_place,
    bearing_word, wrap_deg, NEAR_M, NEXT_TO_M,
)
from .events import (
    BrainEvent, EventLog, event_json, parse_event, load_events, robot_dir,
    EV_SESSION, EV_HEARD, EV_SAID, EV_BEGAN, EV_GOING, EV_ARRIVED,
    EV_MET, EV_MISSED, EV_SPEAKER, EV_TASK_STARTED, EV_STEP_STARTED, EV_STEP_DONE, EV_TASK_DONE, EV_TASK_FAILED, EV_TASK_CANCELLED,
    EV_SAW, EV_TOLD_WHERE, EV_LOST, EV_PICKED, EV_PUT,
)
from .objects import (
    ObjectXY, object_event_text, parse_object_event, heading_mod, heading_diff, where_word,
    OBJ_FLOOR, OBJ_HELD, OBJ_PUT, HOW_SEEN, HOW_TOLD, HOW_REMEMBERED, OBJ_FRESH_S, OBJ_GAP_S, OBJ_MOVED_M, OBJ_TURNED_DEG,
)
from .state import (
    WorldState, Body, Skill, motion_skill, known_motion, PlaceWord, Task, Step, WS_EVENTS_N, WS_TRAIL_N,
    TASK_RUNNING, TASK_DONE, TASK_FAILED, TASK_CANCELLED, STEP_PENDING, STEP_RUNNING, STEP_DONE, STEP_FAILED,
)
from .memory import (
    restore_memory, digest_lines, read_log, log_days, when_words, save_places, load_places, time_refs, focused_memory,
    restore_objects, is_object_event,
    MEM_DAYS, MEM_TASKS, MEM_HEARD,
)
from .people import People, Person, name_given, asks_identity, identity_answer, unit, dot, SURE_COS, MAYBE_COS, VOICES_MAX
from .tasks import (
    ExecAction, give_task, cancel_task, refuse_step, tick_tasks, notice_text,
    EX_NONE, EX_START, EX_NOTICE, STOP_COMMAND, NOTICE_NOT_REACHED, NOTICE_REFUSED, NOTICE_ALL_DONE,
    NOTICE_NO_OBJECT, NOTICE_NOT_DONE, jusqua,
)
from .answers import (
    asks_task_done, task_done_answer, computed_reply, computed_reply_final, computed_turn, ComputedTurn,
    asks_capabilities, capabilities_answer, asks_can, can_turn,
    asks_for_more, reply_sentences, REPLY_SENTENCES, REPLY_SENTENCES_MORE,
)
from .views import ReplyView, reply_view, reply_stable, reply_live, task_sentences, speaker_sentence, memory_section, metres, language_name, REPLY_VIEW_LINES, REPLY_VIEW_SENTENCES
