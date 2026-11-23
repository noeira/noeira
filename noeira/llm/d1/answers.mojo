# +--------------------------------------------------------------------------+ #
# | d1 — typed answers made from their slots' distributions
# +--------------------------------------------------------------------------+ #
"""d1's `api.answer` for the typed questions, and what it reads them with
(`grid.Grid.read`, Python's `math.fsum`), transcribed
(docs/NN_MODEL_PORTS_PLAN.md §12.9).

- A read (`text`, `number`, `date`): its found through `found:<type>`; when
  it is 0.5 or more, the text and span its slots point at (and the JSON path
  of the string it lies in); a number's or a date's value parsed from that
  text in the state's context (`parsing.mojo`).
- A box: presence then four edges, each edge a distribution over the
  thousandths 0-0.999 (or 128 codes over 0-1); `found` through `found:box`,
  the edges at their medians (or, `d1_box_edges: interp`, where the CDF
  crosses one half), crossed edges as one bin about their middle; the
  confidence is the share of 512 Halton draws through the edge CDFs whose box
  has IoU >= 0.5 with it, through `box`. `boxes`: blocks of the same, block k
  present with P(count >= k) through `found:boxes` kept non-increasing, the
  present ones in reading order (left edge, then top). Every product that
  feeds an addition is rounded on its own (`mul_unfused`): LLVM would fuse it
  where Python rounds twice.
- An estimate (`number` with a `range`): its one slot is a distribution over
  the grid's values; the answer is the median and the p10-p90 interval, each
  quantile read at the level the `estimate` table gives it
  (`Calibration.level`), and the unit.
"""

from noeira.io.json import JsonWriter

from .prompt import D1Question, D1Grid, mul_unfused
from .calibration import D1Calibration
from .pointer import D1Read
from .parsing import D1Parser, cps


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


def answer_read(mut w: JsonWriter, q: D1Question, rd: D1Read, cal: D1Calibration, parser: D1Parser,
                state_text: String) raises:
    """`api.answer` for a read: `{"type", <type>: value, "span", "found"[, "text"][, "path"][, "unit"]}`.
    A number's value is `parse_number(text, state_text, unit=q.unit)` and its unit `q.unit or
    written_unit(text)`; a date's value `parse_date(text, state_text)`'s ISO date."""
    var found = cal.through(String("found:") + q.read_type, rd.found)
    var present = found >= 0.5
    var has = present and rd.has_text
    var has_unit = q.has_unit
    var unit = q.unit
    w.key(q.name)
    w.begin_object()
    w.member("type", q.read_type)
    w.key(q.read_type)
    if not has:
        w.null()
    elif q.read_type == "text":
        w.string(rd.text)
    elif q.read_type == "number":
        var t = cps(rd.text)
        if not (q.has_unit and q.unit.byte_length() > 0):  # `unit or written_unit(text)`
            var wu = parser.written_unit(t)
            has_unit = wu[0]
            unit = wu[1]
        var v = parser.parse_number(t, cps(state_text), q.unit, q.has_unit)
        if v.kind == 1:
            w.integer(v.i)
        elif v.kind == 2:
            w.number(v.f)
        else:
            w.null()
    else:
        var d = parser.parse_date(cps(rd.text), cps(state_text))
        if d.found:
            w.string(d.iso)
        else:
            w.null()
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
        if has_unit:
            w.string(unit)
        else:
            w.null()
    w.end_object()


# ── boxes (`box.py`) ─────────────────────────────────────────────────────

comptime DRAWS: Int = 512


def _halton() -> List[List[Float64]]:
    """`box.samples`: DRAWS points of the Halton sequence in bases 2, 3, 5, 7."""
    var out = List[List[Float64]](capacity=DRAWS)
    var bases: List[Int] = [2, 3, 5, 7]
    for i in range(1, DRAWS + 1):
        var pt = List[Float64]()
        for b in bases:
            var n = i
            var value = 0.0
            var scale = 1.0
            while n > 0:
                var digit = n % b
                n = n // b
                scale /= Float64(b)
                value = value + mul_unfused(Float64(digit), scale)
            pt.append(value)
        out.append(pt^)
    return out^


def edge_grid_of(row: List[Float64]) -> D1Grid:
    """`box.read_as`: the thousandths for 1,000 values, else the 128 codes over 0-1."""
    if len(row) == 1000:
        return D1Grid(0.0, 0.999, 1000)
    return D1Grid(0.0, 1.0, 128)


@fieldwise_init
struct Rect(Copyable, Movable):
    var top: Float64
    var bottom: Float64
    var left: Float64
    var right: Float64


def _ordered(a: Float64, b: Float64, g: D1Grid) -> Tuple[Float64, Float64]:
    """`box.ordered`: crossed edges become one bin about their middle."""
    if a < b:
        return (a, b)
    var half = g.step() / 2.0
    var middle = min(1.0 - half, max(half, (a + b) / 2.0))
    return (middle - half, middle + half)


def rectangle(e: List[Float64], g: D1Grid) -> Rect:
    var tb = _ordered(e[0], e[1], g)
    var lr = _ordered(e[2], e[3], g)
    return Rect(tb[0], tb[1], lr[0], lr[1])


def _area(r: Rect) -> Float64:
    return mul_unfused(r.bottom - r.top, r.right - r.left)


def iou(a: Rect, b: Rect) -> Float64:
    var overlap = mul_unfused(max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top)),
                              max(0.0, min(a.right, b.right) - max(a.left, b.left)))
    return overlap / (_area(a) + _area(b) - overlap)


def _median(row: List[Float64], g: D1Grid) -> Float64:
    """`Grid.read(row)["median"]`: the first value whose cumulative share reaches one half."""
    var total = fsum(row)
    var values = g.values()
    var mass = 0.0
    for i in range(len(row)):
        mass += row[i] / total
        if mass >= 0.5:
            return values[i]
    return values[len(values) - 1]


def _crossing(row: List[Float64], g: D1Grid) -> Float64:
    """`box.crossing`: where the CDF crosses one half, each bin's mass spread evenly over it."""
    var total = fsum(row)
    var values = g.values()
    var st = g.step()
    var mass = 0.0
    for i in range(len(row)):
        var share = row[i] / total
        if share > 0.0 and mass + share >= 0.5:
            var x = values[i] - st / 2.0 + mul_unfused((0.5 - mass) / share, st)
            return min(g.high, max(g.low, x))
        mass += share
    return g.high


def medians(edges: List[List[Float64]], edge_read: String) raises -> Rect:
    """`box.medians`: the box at the four edge distributions' medians."""
    var g = edge_grid_of(edges[0])
    var e = List[Float64]()
    for k in range(4):
        if edge_read == "interp":
            e.append(_crossing(edges[k], g))
        elif edge_read == "median":
            e.append(_median(edges[k], g))
        else:
            raise Error("d1: box edges read as median or interp, not " + edge_read)
    return rectangle(e, g)


def confidence(edges: List[List[Float64]], box: Rect, draws: List[List[Float64]]) -> Float64:
    """`box.confidence`: the share of boxes drawn through the edges' CDFs with IoU >= 0.5 against `box`."""
    var g = edge_grid_of(edges[0])
    var values = g.values()
    var cdfs = List[List[Float64]]()
    for k in range(4):
        var total = fsum(edges[k])
        var c = List[Float64](capacity=len(edges[k]))
        var acc = 0.0
        for p in edges[k]:
            acc += p / total
            c.append(acc)
        cdfs.append(c^)
    var hits = 0
    for ref pt in draws:
        var e = List[Float64]()
        for k in range(4):
            ref c = cdfs[k]
            var lo = 0
            var hi = len(c)
            while lo < hi:  # bisect_left
                var mid = (lo + hi) // 2
                if c[mid] < pt[k]:
                    lo = mid + 1
                else:
                    hi = mid
            e.append(values[min(g.size - 1, lo)])
        if iou(rectangle(e, g), box) >= 0.5:
            hits += 1
    return Float64(hits) / Float64(DRAWS)


def _write_rect(mut w: JsonWriter, r: Rect) raises:
    w.begin_object()
    w.member("top", r.top)
    w.member("bottom", r.bottom)
    w.member("left", r.left)
    w.member("right", r.right)
    w.end_object()


def answer_box(mut w: JsonWriter, q: D1Question, slots: List[List[Float64]], cal: D1Calibration, edge_read: String) raises:
    """`box.answer`: `{"type": "box", "box": {top, bottom, left, right} | null, "found", "confidence"}`."""
    var found = cal.through("found:box", slots[0][0])
    w.key(q.name)
    w.begin_object()
    w.member("type", String("box"))
    if found < 0.5:
        w.key("box")
        w.null()
        w.member("found", found)
        w.member("confidence", 0.0)
        w.end_object()
        return
    var edges = List[List[Float64]]()
    for k in range(1, 5):
        edges.append(slots[k].copy())
    var box = medians(edges, edge_read)
    w.key("box")
    _write_rect(w, box)
    w.member("found", found)
    w.member("confidence", cal.through("box", confidence(edges, box, _halton())))
    w.end_object()


def answer_boxes(mut w: JsonWriter, q: D1Question, slots: List[List[Float64]], cal: D1Calibration, edge_read: String) raises:
    """`box.answer_many`: blocks of presence and four edges; presence k =
    P(count >= k) through `found:boxes`, kept non-increasing; the present
    blocks' boxes in reading order (left, then top), each with its found and
    its confidence through `boxes`; `count`, `at_least` (every block read is
    present) and `presence`."""
    var presence = List[Float64]()
    var below = 1.0
    var k = 0
    while k < len(slots):
        below = min(below, cal.through("found:boxes", slots[k][0]))
        presence.append(below)
        k += 5
    var boxes = List[Rect]()
    var founds = List[Float64]()
    var confs = List[Float64]()
    var draws = _halton()
    for b in range(len(presence)):
        if presence[b] < 0.5:
            break
        var edges = List[List[Float64]]()
        for j in range(1, 5):
            edges.append(slots[5 * b + j].copy())
        var box = medians(edges, edge_read)
        confs.append(cal.through("boxes", confidence(edges, box, draws)))
        founds.append(presence[b])
        boxes.append(box^)
    var order = List[Int]()  # a stable sort on (left, top)
    for i in range(len(boxes)):
        var j = len(order)
        while j > 0:
            ref p = boxes[order[j - 1]]
            if p.left > boxes[i].left or (p.left == boxes[i].left and p.top > boxes[i].top):
                j -= 1
            else:
                break
        order.insert(j, i)
    w.key(q.name)
    w.begin_object()
    w.member("type", String("boxes"))
    w.key("boxes")
    w.begin_array()
    for i in order:
        w.begin_object()
        w.key("box")
        _write_rect(w, boxes[i])
        w.member("found", founds[i])
        w.member("confidence", confs[i])
        w.end_object()
    w.end_array()
    w.member("count", len(boxes))
    w.key("at_least")
    w.boolean(len(boxes) > 0 and len(boxes) == len(presence))
    w.key("presence")
    w.begin_array()
    for p in presence:
        w.number(p)
    w.end_array()
    w.end_object()
