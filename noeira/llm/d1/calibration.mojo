# +--------------------------------------------------------------------------+ #
# | d1 — calibration.json: the temperature and the typed answers' tables
# +--------------------------------------------------------------------------+ #
"""d1's `calibration.py` (docs/NN_MODEL_PORTS_PLAN.md §12.9).

- `default`: the temperature an untyped question's option scores are divided
  by (`noul`, `choice`, `score`); per-bucket temperatures are refused (d1-2B
  has none).
- `tables`: monotone (raw, calibrated) knots a typed answer's probability goes
  through, linear between knots, flat past the ends (`calibration.through`):
  `found:text|number|date`, `found:box`, `box`, `found:boxes`, `boxes`,
  `multiple`, and `estimate` (a quantile's level, read backwards by `level`).
  A typed slot's softmax is NOT tempered: the tables were fitted on raw reads.
"""

from noeira.io.json import load_json, J_OBJECT


struct D1Calibration(Copyable, Movable):
    var default: Float64
    var names: List[String]
    var raw: List[List[Float64]]
    var cal: List[List[Float64]]

    def __init__(out self):
        self.default = 1.0
        self.names = List[String]()
        self.raw = List[List[Float64]]()
        self.cal = List[List[Float64]]()

    @staticmethod
    def load(path: String) raises -> D1Calibration:
        var doc = load_json(path)
        var r = doc.root()
        var c = D1Calibration()
        var d = doc.field(r, "default")
        if d >= 0:
            c.default = doc.number(d)
        var per = doc.field(r, "temperatures")
        if per >= 0 and doc.size(per) > 0:
            raise Error("d1: per-bucket temperatures are not implemented")
        var t = doc.field(r, "tables")
        if t >= 0 and doc.kind_of(t) == J_OBJECT:
            for i in range(doc.size(t)):
                c.names.append(doc.key_at(t, i))
                var knots = doc.at(t, i)
                var xs = List[Float64]()
                var ys = List[Float64]()
                for k in range(doc.size(knots)):
                    var kn = doc.at(knots, k)
                    xs.append(doc.number(doc.at(kn, 0)))
                    ys.append(doc.number(doc.at(kn, 1)))
                c.raw.append(xs^)
                c.cal.append(ys^)
        return c^

    def _index(self, name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1

    def has(self, name: String) -> Bool:
        return self._index(name) >= 0

    def through(self, name: String, value: Float64) -> Float64:
        """`Calibration.through`: `value` through the table `name`; unchanged
        without one."""
        var i = self._index(name)
        if i < 0:
            return value
        return _through(value, self.raw[i], self.cal[i])

    def level(self, name: String, share: Float64) -> Float64:
        """`Calibration.level`: the level whose observed share is `share` (the
        table read backwards); `share` itself without one."""
        var i = self._index(name)
        if i < 0:
            return share
        return _through(share, self.cal[i], self.raw[i])


def _through(value: Float64, xs: List[Float64], ys: List[Float64]) -> Float64:
    """`calibration.through`: bisect_left on the raws, linear between knots,
    flat past the ends."""
    var n = len(xs)
    if n == 0:
        return value
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) // 2
        if xs[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    if lo == 0:
        return ys[0]
    if lo == n:
        return ys[n - 1]
    var a = xs[lo - 1]
    var pa = ys[lo - 1]
    var b = xs[lo]
    var pb = ys[lo]
    return pa + (value - a) * (pb - pa) / (b - a)
