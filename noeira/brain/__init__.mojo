"""The robot's brain (docs/ROBOT_BRAIN_PLAN.md): one shared state, its record,
and the text each model reads from it.

    from noeira.brain import WorldState, Body, PlaceXY, reply_view, robot_dir

Step B1: `WorldState` (what the robot is, where it is, what it is doing, what
happened), `EventLog` (the append-only record on disk), `reply_view` (the reply
model's prompt, written from the state) and the place geometry the decision
model's lines are made of. Skills, the task stack and the executive (B2),
persistence across restarts (B3), people (B4) and a second body (B5) are not
here yet.
"""

from .geometry import (
    PlaceXY, PlaceAround, places_around, place_line, place_lines, nearest_place,
    bearing_word, wrap_deg, NEAR_M, NEXT_TO_M,
)
from .events import (
    BrainEvent, EventLog, event_json, parse_event, load_events, robot_dir,
    EV_SESSION, EV_HEARD, EV_SAID, EV_BEGAN, EV_GOING, EV_ARRIVED,
)
from .state import WorldState, Body, PlaceWord, WS_EVENTS_N, WS_TRAIL_N
from .views import ReplyView, reply_view, reply_stable, reply_live, metres, language_name, REPLY_VIEW_LINES, REPLY_VIEW_SENTENCES
