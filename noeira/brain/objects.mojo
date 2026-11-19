# +--------------------------------------------------------------------------+ #
# | Things the robot can move: one record each, where it is believed to be now
# +--------------------------------------------------------------------------+ #
"""A movable object of the room in the brain's state: the crate the robot
carries (docs/ROBOT_BRAIN_PLAN.md §5.1 "places and objects", §20).

    ws.add_object(ObjectXY.box("crate", "crate", 0.13, 0.215, 0.19))  # its size, no place yet
    ws.told_object("crate", 2.0, 0.5, 30.0, "trials", t, now)         # where it is said to be
    ws.saw_object("crate", x, y, yaw_deg, n_px, "crate_eye", t, now)  # a perception fit
    ws.picked_up("crate", t, now)                                     # the hands have it
    ws.put_down("crate", "table", x, y, yaw_deg, t, now)              # it was put at a place
    ws.object_missing("crate", t, now)                                # looked there: not there

    var i = ws.object_index("crate")      # -1 = not declared
    ws.objects[i].x, .y, .yaw_deg, .where, .put_at, .how, .fresh(t)

⚠ ONE RECORD PER OBJECT, REPLACED, NOT ACCUMULATED. A fixture's place is
accumulated over a tour (`noeira.perception.places`); a thing that is picked
up and put down elsewhere would leave a stale copy of itself behind that way.
Each report REPLACES the pose, and the record says how the pose is known
(`how`: seen, told, remembered) and when (`t_seen` on the caller's clock,
`now_seen` in epoch seconds).

⚠ THE HEADING IS MODULO `turn_deg`. A box has no front: its heading is that of
its own x axis, modulo a half turn (`turn_deg` = 180), a quarter turn when
its footprint is square (90). It is kept in [0, `turn_deg`); `heading_near`
gives the representative nearest a wanted heading. Degrees, counter-clockwise
from the room's +x, as the robot's own `yaw_deg`.

⚠ THE STATE IS REPORTED MANY TIMES A SECOND, THE LOG IS NOT. A tracker may
call `saw_object` at every control step while the object is in view. The
record takes every report; a `saw` event is written only when the sighting is
news: the first one, the first after `OBJ_GAP_S` without one, the first after
it was missing or only told or remembered, or a move of more than
`OBJ_MOVED_M` / `OBJ_TURNED_DEG` from the pose last written. While the object
is HELD, its pose follows the hands and nothing is written until it is put
down.

THE CALLER REPORTS FACTS: whatever decides that the hands have the object
(the object policy, a grasp check) calls `picked_up`; whatever releases it
calls `put_down` with the place it was put at ("" = on the floor, at no
place). The executive's `pick:` and `put:` steps end on those facts
(`noeira.brain.tasks`).

THE LOG KEEPS THE LAST POSE: each object event's text is the whole record
(`object_event_text`), so `restore_objects` (`noeira.brain.memory`) rebuilds
the objects of earlier sessions as `remembered`.
"""

from std.math import floor


comptime OBJ_FLOOR: Int = 0
"""On the floor (or wherever it was found), at no particular place."""
comptime OBJ_HELD: Int = 1
"""In the robot's hands."""
comptime OBJ_PUT: Int = 2
"""Put down by the robot at a place (`put_at`)."""

comptime HOW_SEEN = "seen"
"""The pose is a perception fit."""
comptime HOW_TOLD = "told"
"""The pose was given: a trials table, a person, a scene file."""
comptime HOW_REMEMBERED = "remembered"
"""The pose is the last one an earlier session wrote in the log."""

comptime OBJ_FRESH_S: Float64 = 0.5
"""A sighting younger than this is `fresh`: the object is in view now."""
comptime OBJ_GAP_S: Float64 = 2.0
"""A sighting after this long without one is written to the log."""
comptime OBJ_MOVED_M: Float64 = 0.10
"""A sighting this far from the pose last written is written (the object moved)."""
comptime OBJ_TURNED_DEG: Float64 = 10.0
"""A sighting turned this much from the heading last written is written."""


def heading_mod(yaw_deg: Float64, turn_deg: Float64) -> Float64:
    """`yaw_deg` into [0, `turn_deg`)."""
    var r = yaw_deg - turn_deg * floor(yaw_deg / turn_deg)
    if r >= turn_deg:
        r -= turn_deg
    return r


def heading_diff(a_deg: Float64, b_deg: Float64, turn_deg: Float64) -> Float64:
    """The smallest turn between two headings of an object whose heading is
    modulo `turn_deg`, in [0, `turn_deg` / 2]."""
    var d = heading_mod(a_deg - b_deg, turn_deg)
    return min(d, turn_deg - d)


def where_word(where: Int) -> String:
    """`floor`, `held` or `put`: the log's word for `where`."""
    if where == OBJ_HELD:
        return String("held")
    if where == OBJ_PUT:
        return String("put")
    return String("floor")


def where_from_word(word: String) -> Int:
    if word == "held":
        return OBJ_HELD
    if word == "put":
        return OBJ_PUT
    return OBJ_FLOOR


struct ObjectXY(Copyable, Movable):
    """One movable object: what it is, its size, where it is believed to be."""

    var name: String
    """Its id in the state, unique: `crate`, `crate_2`. Also the key of its
    words in people's languages (`WorldState.call_place`)."""
    var label: String
    """What kind of thing it is: `crate`, `box`."""
    var hx: Float64
    """Half sizes along its own x, y and z, metres."""
    var hy: Float64
    var hz: Float64
    var turn_deg: Float64
    """Its heading is modulo this: 180 for a box, 90 for a square footprint,
    360 for a thing with a front."""

    var has_pose: Bool
    """A pose was reported or remembered; False = declared only."""
    var x: Float64
    """Its centre on the floor, the room's frame, metres."""
    var y: Float64
    var yaw_deg: Float64
    """The heading of its own x axis, in [0, `turn_deg`)."""
    var where: Int
    """`OBJ_FLOOR`, `OBJ_HELD` or `OBJ_PUT`."""
    var put_at: String
    """The place it was put down at (`OBJ_PUT`); "" otherwise."""
    var how: String
    """`HOW_SEEN`, `HOW_TOLD` or `HOW_REMEMBERED`; "" before any pose."""
    var source: String
    """Who reported the pose: `crate_eye`, `trials`, `log`."""
    var score: Float64
    """The last sighting's own measure, as its source gives it (the crate's
    eye: labelled pixels). 0 for a pose that was not seen."""
    var missing: Bool
    """The robot looked where the record says and did not find it
    (`object_missing`); a sighting clears it."""
    var t_seen: Float64
    """The caller's clock (seconds) at the last report in THIS process; -1 = none."""
    var now_seen: Int
    """Epoch seconds of the last report (a remembered pose: its event's)."""
    var times_seen: Int
    """Sightings in this process."""

    var t_logged: Float64
    """The caller's clock at the last pose written to the log; -1 = none."""
    var lx: Float64
    """The pose last written to the log (`OBJ_MOVED_M`, `OBJ_TURNED_DEG`)."""
    var ly: Float64
    var lyaw: Float64

    def __init__(out self, var name: String, var label: String,
                 hx: Float64, hy: Float64, hz: Float64, turn_deg: Float64):
        self.name = name^
        self.label = label^
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.turn_deg = turn_deg
        self.has_pose = False
        self.x = 0.0
        self.y = 0.0
        self.yaw_deg = 0.0
        self.where = OBJ_FLOOR
        self.put_at = String("")
        self.how = String("")
        self.source = String("")
        self.score = 0.0
        self.missing = False
        self.t_seen = -1.0
        self.now_seen = 0
        self.times_seen = 0
        self.t_logged = -1.0
        self.lx = 0.0
        self.ly = 0.0
        self.lyaw = 0.0

    @staticmethod
    def box(name: String, label: String, hx: Float64, hy: Float64, hz: Float64) -> ObjectXY:
        """A box of half sizes (hx, hy, hz): heading modulo a half turn, a
        quarter turn when its footprint is square."""
        return ObjectXY(name.copy(), label.copy(), hx, hy, hz, 90.0 if hx == hy else 180.0)

    def fresh(self, t: Float64) -> Bool:
        """Seen in this process within `OBJ_FRESH_S` of `t`: in view now."""
        return self.how == HOW_SEEN and self.t_seen >= 0.0 and t - self.t_seen <= OBJ_FRESH_S

    def heading_near(self, want_deg: Float64) -> Float64:
        """Its heading, as the representative nearest `want_deg`: in
        (want − turn/2, want + turn/2]."""
        var d = heading_mod(self.yaw_deg - want_deg, self.turn_deg)
        if d > 0.5 * self.turn_deg:
            d -= self.turn_deg
        return want_deg + d

    def set_pose(mut self, x: Float64, y: Float64, yaw_deg: Float64):
        self.has_pose = True
        self.x = x
        self.y = y
        self.yaw_deg = heading_mod(yaw_deg, self.turn_deg)

    def moved_since_logged(self) -> Bool:
        """The pose is more than `OBJ_MOVED_M` or `OBJ_TURNED_DEG` from the one last written."""
        var dx = self.x - self.lx
        var dy = self.y - self.ly
        return (dx * dx + dy * dy > OBJ_MOVED_M * OBJ_MOVED_M
                or heading_diff(self.yaw_deg, self.lyaw, self.turn_deg) > OBJ_TURNED_DEG)

    def mark_logged(mut self, t: Float64):
        self.t_logged = t
        self.lx = self.x
        self.ly = self.y
        self.lyaw = self.yaw_deg


def _fixed(v: Float64, digits: Int) -> String:
    return String(round(v, digits))


def object_event_text(o: ObjectXY) -> String:
    """The whole record as an event's text, read back by `parse_object_event`:

        crate | 1.234 0.456 35.0 | put table | crate 0.13 0.215 0.19 180.0 | seen

    ⚠ ONE FORMAT, WRITTEN AND READ HERE: a second writer would drift."""
    var where = where_word(o.where)
    if o.where == OBJ_PUT:
        where += " " + o.put_at
    return (
        o.name + " | " + _fixed(o.x, 3) + " " + _fixed(o.y, 3) + " " + _fixed(o.yaw_deg, 1)
        + " | " + where
        + " | " + o.label + " " + _fixed(o.hx, 3) + " " + _fixed(o.hy, 3) + " " + _fixed(o.hz, 3)
        + " " + _fixed(o.turn_deg, 1)
        + " | " + o.how
    )


def parse_object_event(text: String) raises -> ObjectXY:
    """An object event's text back as a record with its pose (`has_pose`),
    `how` as it was written. Raises on a text of another shape."""
    var f = text.split(" | ")
    if len(f) != 5:
        raise Error("object event: 5 fields expected, got " + String(len(f)) + ": " + text)
    var pose = String(f[1]).split(" ")
    var where = String(f[2]).split(" ")
    var kind = String(f[3]).split(" ")
    if len(pose) != 3 or len(where) < 1 or len(kind) != 5:
        raise Error("object event: malformed: " + text)
    var o = ObjectXY(
        String(f[0]), String(kind[0]),
        Float64(String(kind[1])), Float64(String(kind[2])), Float64(String(kind[3])), Float64(String(kind[4])),
    )
    o.set_pose(Float64(String(pose[0])), Float64(String(pose[1])), Float64(String(pose[2])))
    o.where = where_from_word(String(where[0]))
    if o.where == OBJ_PUT and len(where) > 1:
        var place = String("")
        for i in range(1, len(where)):
            if i > 1:
                place += " "
            place += String(where[i])
        o.put_at = place^
    o.how = String(f[4])
    return o^
