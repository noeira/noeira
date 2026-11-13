# +--------------------------------------------------------------------------+ #
# | Places around a robot, in words: bearing bucket, rounded distance, neighbour
# +--------------------------------------------------------------------------+ #
"""Where each known place is, FROM THE ROBOT, as a model can read it.

    var around = places_around(places, x, y, yaw_deg)     # one PlaceAround per place
    place_line(around[0])                                 # "table: left, 3.5 m, next to sofa"

⚠ THE ARITHMETIC STAYS ON THIS SIDE. A model is given buckets and words, never
coordinates to subtract: bearings are one of four words, distances are rounded
to 0.5 m. Jev's own notes list arithmetic and numeric comparison as failure
modes, and RB-1 measured the compact line below as the largest single effect
on spatial requests (19 -> 48 of 70; `docs/ROOM_BENCH_RB1_PLAN.md`).

⚠ ONE COPY. RB-1's bench state (`projects/g1/scripts/rb1_state.mojo`, private)
imports these under its own names; its gate against the Python reference that
froze the bench's labels (`test_rb1_state.mojo`) is therefore also this
module's. The brain's own gate is `tests/brain/test_brain_state.mojo`, against
`tools/brain/make_bb1.py`'s independent arithmetic.

Frame: x, y in metres in the room's frame; yaw in degrees, 0 = +x,
counter-clockwise. `left` is a positive relative bearing.
"""

from std.math import sqrt, atan2, floor, pi


comptime NEAR_M = 2.5
"""Closer than this is `near`."""
comptime NEXT_TO_M = 3.0
"""A place's `next_to` is its nearest OTHER place, when within this."""


@fieldwise_init
struct PlaceXY(Copyable, Movable):
    """A place's name and centre in the room's frame."""

    var name: String
    var x: Float64
    var y: Float64


@fieldwise_init
struct PlaceAround(Copyable, Movable):
    """One place as seen from the robot."""

    var name: String
    var bearing: String
    """`ahead`, `left`, `right` or `behind`: 90-degree sectors about the heading."""
    var distance_m: Float64
    """Rounded to 0.5 m."""
    var range: String
    """`near` (under `NEAR_M`, on the unrounded distance) or `far`."""
    var next_to: String
    """The nearest other place within `NEXT_TO_M` of this one, or `none`."""


def wrap_deg(a: Float64) -> Float64:
    """Degrees into (-180, 180]."""
    var r = a - 360.0 * floor(a / 360.0)
    if r > 180.0:
        r -= 360.0
    return r


def bearing_word(rel: Float64) -> String:
    """The sector of a relative bearing in degrees (left positive). The edges:
    +-45 are `ahead`, +135 is `left`, -135 is `right`."""
    if abs(rel) <= 45.0:
        return String("ahead")
    if rel > 45.0 and rel <= 135.0:
        return String("left")
    if rel >= -135.0 and rel < -45.0:
        return String("right")
    return String("behind")


def places_around(entries: List[PlaceXY], x: Float64, y: Float64, yaw: Float64) -> List[PlaceAround]:
    """Per entry, in the entries' order: bearing bucket, distance (0.5 m),
    near/far, nearest OTHER entry within 3 m."""
    var out = List[PlaceAround]()
    for i in range(len(entries)):
        var dx = entries[i].x - x
        var dy = entries[i].y - y
        var d = sqrt(dx * dx + dy * dy)
        var rel = wrap_deg(atan2(dy, dx) * 180.0 / pi - yaw)
        var best = String("none")
        var bd = 1e9
        for j in range(len(entries)):
            if j == i:
                continue
            var ex = entries[j].x - entries[i].x
            var ey = entries[j].y - entries[i].y
            var dd = sqrt(ex * ex + ey * ey)
            if dd < bd:
                bd = dd
                best = entries[j].name
        out.append(PlaceAround(
            entries[i].name,
            bearing_word(rel),
            floor(d * 2.0 + 0.5) / 2.0,
            String("near") if d < NEAR_M else String("far"),
            best if bd <= NEXT_TO_M else String("none"),
        ))
    return out^


def place_line(p: PlaceAround) -> String:
    """One place, compact: `"table: left, 3.5 m"` (`, next to X` only when
    there is one). ⚠ The one copy of the format d1's decisions were measured
    with (RB-1's `s1c`): the bench and `G1Context.around` both write it here."""
    var t = p.name + ": " + p.bearing + ", " + String(p.distance_m) + " m"
    if p.next_to != "none":
        t += ", next to " + p.next_to
    return t


def place_lines(entries: List[PlaceXY], x: Float64, y: Float64, yaw_deg: Float64) -> List[String]:
    """`G1Context.around` for a robot at (x, y), heading `yaw_deg`."""
    var out = List[String]()
    for p in places_around(entries, x, y, yaw_deg):
        out.append(place_line(p))
    return out^


def nearest_place(entries: List[PlaceXY], x: Float64, y: Float64) -> String:
    """The closest entry's name on the UNROUNDED distance ("" for no entry;
    the first of equals)."""
    var best = String("")
    var bd = 1e18
    for e in entries:
        var d = (e.x - x) * (e.x - x) + (e.y - y) * (e.y - y)
        if d < bd:
            bd = d
            best = e.name
    return best
