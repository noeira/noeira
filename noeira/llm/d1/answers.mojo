# +--------------------------------------------------------------------------+ #
# | d1 — typed answers made from their slots' distributions
# +--------------------------------------------------------------------------+ #
"""d1's `api.answer` for the typed questions, and what it reads them with
(`grid.Grid.read`, Python's `math.fsum`), transcribed
(docs/NN_MODEL_PORTS_PLAN.md §12.9).

- A read (`text`, `number`, `date`): its found through `found:<type>`; when
  it is 0.5 or more, the text and span its slots point at (and the JSON path
  of the string it lies in). A number's or a date's value needs d1's parser
  (T6): null until then.
- An estimate (`number` with a `range`): its one slot is a distribution over
  the grid's values; the answer is the median and the p10-p90 interval, each
  quantile read at the level the `estimate` table gives it
  (`Calibration.level`), and the unit.
"""

from noeira.io.json import JsonWriter

from .prompt import D1Question, D1Grid
from .calibration import D1Calibration
from .pointer import D1Read


def fsum(xs: List[Float64]) -> Float64:
    """Python's `math.fsum`: the correctly rounded sum (Shewchuk's partials,
    with CPython's final half-way correction)."""
    var partials = List[Float64]()
    for x0 in xs:
        var x = x0
        var i = 0
        for j in range(len(partials)):
            var y = partials[j]
            if abs(x) < abs(y):
                var t = x
                x = y
                y = t
            var hi = x + y
            var lo = y - (hi - x)
            if lo != 0.0:
                partials[i] = lo
                i += 1
            x = hi
        while len(partials) > i:
            _ = partials.pop()
        partials.append(x)
    var n = len(partials)
    if n == 0:
        return 0.0
    n -= 1
    var hi = partials[n]
    var lo = 0.0
    while n > 0:
        var x = hi
        n -= 1
        var y = partials[n]
        hi = x + y
        var yr = hi - x
        lo = y - yr
        if lo != 0.0:
            break
    if n > 0 and ((lo < 0.0 and partials[n - 1] < 0.0) or (lo > 0.0 and partials[n - 1] > 0.0)):
        var y = lo * 2.0
        var x = hi + y
        var yr = x - hi
        if y == yr:
            hi = x
    return hi


@fieldwise_init
struct GridRead(Copyable, Movable):
    var mean: Float64
    var p10: Float64
    var median: Float64
    var p90: Float64


def grid_read(g: D1Grid, probs: List[Float64], cal: D1Calibration) raises -> GridRead:
    """`Grid.read(probs, level)` with `level = cal.level("estimate", q)`: the
    mean, and the first value whose cumulative share reaches each quantile's
    level (the last value for a level the summed mass falls short of)."""
    if len(probs) != g.size:
        raise Error("d1: an estimate's distribution has " + String(len(probs)) + " values, its grid " + String(g.size))
    var total = fsum(probs)
    var values = g.values()
    var terms = List[Float64](capacity=g.size)
    for i in range(g.size):
        terms.append(values[i] * (probs[i] / total))
    var mean = fsum(terms)
    var lv10 = cal.level("estimate", 0.1)
    var lv50 = cal.level("estimate", 0.5)
    var lv90 = cal.level("estimate", 0.9)
    var got10 = False
    var got50 = False
    var got90 = False
    var p10 = values[g.size - 1]
    var p50 = values[g.size - 1]
    var p90 = values[g.size - 1]
    var mass = 0.0
    for i in range(g.size):
        mass += probs[i] / total
        if not got10 and mass >= lv10:
            p10 = values[i]
            got10 = True
        if not got50 and mass >= lv50:
            p50 = values[i]
            got50 = True
        if not got90 and mass >= lv90:
            p90 = values[i]
            got90 = True
    return GridRead(mean, p10, p50, p90)


def answer_estimate(mut w: JsonWriter, q: D1Question, slots: List[List[Float64]], cal: D1Calibration) raises:
    """`api.answer` for an estimate: `{"type": "number", "number": median,
    "interval": [p10, p90], "unit": unit or null}`."""
    var r = grid_read(q.grid, slots[0], cal)
    w.key(q.name)
    w.begin_object()
    w.member("type", String("number"))
    w.member("number", r.median)
    w.key("interval")
    w.begin_array()
    w.number(r.p10)
    w.number(r.p90)
    w.end_array()
    w.key("unit")
    if q.has_unit:
        w.string(q.unit)
    else:
        w.null()
    w.end_object()


def answer_read(mut w: JsonWriter, q: D1Question, rd: D1Read, cal: D1Calibration) raises:
    """`api.answer` for a read: `{"type", <type>: value, "span", "found"[, "text"][, "path"][, "unit"]}`."""
    var found = cal.through(String("found:") + q.read_type, rd.found)
    var present = found >= 0.5
    var has = present and rd.has_text
    w.key(q.name)
    w.begin_object()
    w.member("type", q.read_type)
    w.key(q.read_type)
    if has and q.read_type == "text":
        w.string(rd.text)
    else:
        w.null()  # a number's or a date's value: d1's parser (T6)
    w.key("span")
    if has:
        w.begin_array()
        w.integer(rd.start)
        w.integer(rd.end)
        w.end_array()
    else:
        w.null()
    w.member("found", found)
    if q.read_type != "text":
        w.key("text")
        if has:
            w.string(rd.text)
        else:
            w.null()
    if present and rd.has_path:
        w.member("path", rd.path)
    if q.read_type == "number":
        w.key("unit")
        if q.has_unit:
            w.string(q.unit)
        else:
            w.null()
    w.end_object()
