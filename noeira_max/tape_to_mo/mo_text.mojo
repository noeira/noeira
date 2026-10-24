"""A textual MO-dialect graph, written the way MAX's Python Graph API prints
one (`str(graph)`), so MAX's C API compiles it with no Python in the process
(`M_setModelPath` on a file whose text holds `mo.graph`).

The format is not a supported MAX contract ("the Graph API is the only IR
producer"); `test_tape_to_mo.mojo` gates it against MAX 26.7. Syntax taken
from `str(graph)` of the Python API:

    %3 = rmo.add(%1, %2) : (!mo.tensor<[4, 2], f32>, !mo.tensor<[1, 2], f32>) -> !mo.tensor<[4, 2], f32>
    %4 = rmo.mo.tanh(%3) : (T) -> T
    %5 = rmo.mo.reduce.add(%4) [axis = 0] : (T) -> T'
    %t, %c = rmo.mo.mutable.load(%arg3, %c0) : (!mo.buffer<…>, !mo.chain) -> (!mo.tensor<…>, !mo.chain)
"""


def shape_ty(shape: List[Int], dtype: String = "f32") -> String:
    var s = String("!mo.tensor<[")
    for i in range(len(shape)):
        if i > 0:
            s += ", "
        s += String(shape[i])
    return s + "], " + dtype + ">"


def buffer_ty(shape: List[Int]) -> String:
    var s = String("!mo.buffer<[")
    for i in range(len(shape)):
        if i > 0:
            s += ", "
        s += String(shape[i])
    return s + "], f32>"


def fmt_f32(c: Float32) -> String:
    """The exact f32 as an MLIR float literal (the shortest round-trip
    decimal of the value, which parses back to the same f32). MLIR needs a
    '.' in a float literal: `2` -> `2.0`, `1e-08` -> `1.0e-08`."""
    var s = String(Float64(c))
    if s.find(".") >= 0 or s.find("inf") >= 0 or s.find("nan") >= 0:
        return s
    var e = s.find("e")
    if e < 0:
        return s + ".0"
    return String(s[byte=0:e]) + ".0" + String(s[byte=e:])


struct MoGraph(Movable):
    """SSA values are `%vN`; arguments `%argN`; one chain threaded through
    every buffer load and store, in emission order (or one per buffer)."""

    var name: String
    var arg_types: List[String]
    var arg_names: List[String]
    var out_ids: List[String]
    var out_types: List[String]
    var out_names: List[String]
    var body: String
    var n: Int
    var chain: String
    var per_buffer: Bool
    """Each buffer gets its own chain (`mo.chain.create()` at its first
    load), so its load and store are ordered only against each other."""
    var buf_chains: Dict[String, String]

    def __init__(out self, name: String, per_buffer: Bool = False):
        self.name = name
        self.arg_types = List[String]()
        self.arg_names = List[String]()
        self.out_ids = List[String]()
        self.out_types = List[String]()
        self.out_names = List[String]()
        self.body = String("")
        self.n = 0
        self.chain = String("%chain0")
        self.per_buffer = per_buffer
        self.buf_chains = Dict[String, String]()

    def _v(mut self) -> String:
        var s = String("%v", self.n)
        self.n += 1
        return s

    def _line(mut self, s: String):
        self.body += "  " + s + "\n"

    def arg(mut self, name: String, ty: String) -> String:
        var id = String("%arg", len(self.arg_types))
        self.arg_types.append(ty)
        self.arg_names.append(name)
        return id

    def const(mut self, c: Float32) -> String:
        var v = self._v()
        self._line(String(v, " = mo.constant <#M.dense_array<", fmt_f32(c),
                          "> : tensor<f32>> : !mo.tensor<[], f32>"))
        return v

    def binary(mut self, op: String, a: String, ta: String, b: String, tb: String, tout: String) -> String:
        """`op` in add, sub, mul, div, matmul, greater (rmo.<op>)."""
        var v = self._v()
        self._line(String(v, " = rmo.", op, "(", a, ", ", b, ") : (", ta, ", ", tb, ") -> ", tout))
        return v

    def unary(mut self, op: String, a: String, ta: String, tout: String) -> String:
        """`op` in relu, tanh, sigmoid, exp, log, sqrt (rmo.mo.<op>)."""
        var v = self._v()
        self._line(String(v, " = rmo.mo.", op, "(", a, ") : (", ta, ") -> ", tout))
        return v

    def reduce(mut self, kind: String, a: String, ta: String, axis: Int, tout: String) -> String:
        """`kind` in add, mean; the reduced axis is kept (size 1)."""
        var v = self._v()
        self._line(String(v, " = rmo.mo.reduce.", kind, "(", a, ") [axis = ", axis, "] : (",
                          ta, ") -> ", tout))
        return v

    def transpose2(mut self, a: String, ta: String, tout: String) -> String:
        var perm = self._v()
        self._line(String(perm, " = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>"))
        var v = self._v()
        self._line(String(v, " = rmo.mo.transpose(", a, ", ", perm, ") : (", ta,
                          ", !mo.tensor<[2], si64>) -> ", tout))
        return v

    def broadcast_to(mut self, a: String, ta: String, shape: List[Int]) -> String:
        var dims = String("")
        for i in range(len(shape)):
            if i > 0:
                dims += ", "
            dims += String(shape[i])
        var v = self._v()
        self._line(String(v, " = rmo.broadcast_to(", a, ") [newShape = #mosh<ape[", dims,
                          "]> : !mosh.ape] : (", ta, ") -> ", shape_ty(shape)))
        return v

    def select(mut self, c: String, tc: String, x: String, tx: String, y: String, ty: String, tout: String) -> String:
        var v = self._v()
        self._line(String(v, " = rmo.select(", c, ", ", x, ", ", y, ") : (", tc, ", ", tx, ", ",
                          ty, ") -> ", tout))
        return v

    def _chain_of(mut self, buf: String) -> String:
        if not self.per_buffer:
            return self.chain
        var c = self.buf_chains.get(buf)
        if c:
            return c.value()
        var fresh = String("%bchain", self.n)
        self.n += 1
        self._line(String(fresh, " = mo.chain.create()"))
        return fresh

    def _set_chain(mut self, buf: String, c: String):
        if self.per_buffer:
            self.buf_chains[buf] = c
        else:
            self.chain = c

    def slice0(mut self, a: String, ta: String, start: Int, stop: Int, tout: String) -> String:
        """`a[start:stop]` along axis 0."""
        var v = self._v()
        self._line(String(v, " = rmo.slice(", a, ") [starts = #mosh<ape[", start,
                          "]> : !mosh.ape, stops = #mosh<ape[", stop, "]> : !mosh.ape, steps = #mosh<ape[1]> : !mosh.ape] : (",
                          ta, ") -> ", tout))
        return v

    def reshape(mut self, a: String, ta: String, shape: List[Int]) -> String:
        var dims = String("")
        for i in range(len(shape)):
            if i > 0:
                dims += ", "
            dims += String(shape[i])
        var v = self._v()
        self._line(String(v, " = rmo.reshape(", a, ") [newShape = #mosh<ape[", dims,
                          "]> : !mosh.ape] : (", ta, ") -> ", shape_ty(shape)))
        return v

    def concat0(mut self, ids: List[String], types: List[String], tout: String) -> String:
        var v = self._v()
        var args = String("")
        var tys = String("")
        for i in range(len(ids)):
            if i > 0:
                args += ", "
                tys += ", "
            args += ids[i]
            tys += types[i]
        self._line(String(v, " = rmo.concat(", args, ") [axis = 0 : index] : (", tys, ") -> ", tout))
        return v

    def load(mut self, buf: String, shape: List[Int]) -> String:
        var prev = self._chain_of(buf)
        var t = self._v()
        var c = String("%chain", self.n)
        self.n += 1
        self._line(String(t, ", ", c, " = rmo.mo.mutable.load(", buf, ", ", prev, ") : (",
                          buffer_ty(shape), ", !mo.chain) -> (", shape_ty(shape), ", !mo.chain)"))
        self._set_chain(buf, c)
        return t

    def store(mut self, buf: String, value: String, shape: List[Int]):
        var prev = self._chain_of(buf)
        var c = String("%chain", self.n)
        self.n += 1
        self._line(String(c, " = rmo.mo.mutable.store(", buf, ", ", value, ", ", prev, ") : (",
                          buffer_ty(shape), ", ", shape_ty(shape), ", !mo.chain) -> !mo.chain"))
        self._set_chain(buf, c)

    def output(mut self, id: String, ty: String, name: String):
        self.out_ids.append(id)
        self.out_types.append(ty)
        self.out_names.append(name)

    def text(self) -> String:
        var s = String("mo.graph @", self.name, "(")
        for i in range(len(self.arg_types)):
            if i > 0:
                s += ", "
            s += String("%arg", i, ": ", self.arg_types[i])
        s += ") -> "
        var many = len(self.out_types) != 1
        if many:
            s += "("
        for i in range(len(self.out_types)):
            if i > 0:
                s += ", "
            s += self.out_types[i]
        if many:
            s += ")"
        s += " attributes {_kernel_library_paths = [], argument_names = ["
        for i in range(len(self.arg_names)):
            if i > 0:
                s += ", "
            s += "\"" + self.arg_names[i] + "\""
        s += "], result_names = ["
        for i in range(len(self.out_names)):
            if i > 0:
                s += ", "
            s += "\"" + self.out_names[i] + "\""
        s += "]} {\n  %chain0 = mo.chain.create()\n" + self.body + "  mo.output "
        for i in range(len(self.out_ids)):
            if i > 0:
                s += ", "
            s += self.out_ids[i]
        s += " : "
        for i in range(len(self.out_types)):
            if i > 0:
                s += ", "
            s += self.out_types[i]
        return s + "\n}\n"
