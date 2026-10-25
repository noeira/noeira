"""Functional tape -> one MIL program for the Apple Neural Engine: the
recorded forward and its backward, giving the loss and every parameter's
gradient. The optimizer stays on the host (fp32 master weights).

    var step = emit_grad_step(tape, loss)   # after ONE eager forward
    # arguments, in order (and in name order: i000, i001, ...):
    #   step.args[k] = the tape node fed to argument k (parameters, then the
    #   data inputs), and last the loss-scale seed `[1, 1]`
    # outputs, in order (o000, o001, ...): the loss, then
    #   step.grads[k] = the parameter whose gradient TIMES THE SEED is
    #   output k + 1

Every tensor is fp16 (the M1's ANE refuses fp32 I/O), rank 4 in matrix form:
the tape's `[R, K]` is `[1, 1, R, K]`. The seed is an input so the loss scale
can change without recompiling: fp16's smallest normal is 6.1e-5, below most
gradients of a mean loss. When the loss IS a mean, its 1/count stays on the
host (`grad_div`): the effective scale, seed * count, can then exceed
fp16's 65504, which a batch of 1024 x 64 outputs needs.

Rank-2 tapes only, the ops an MLP and its loss record: add/sub/mul/div with
broadcasting, matmul, scalar ops, relu/tanh/sigmoid/exp/log/sqrt, sum/mean,
transpose, sum_rows, and `F.linear` with or without an activation. The VJPs
are `noeira_max/tape_to_mo/emit.mojo`'s, written as MIL. Anything else raises,
naming the op.
"""

from noeira.experimental.functional import Tape, FTensor
from noeira.experimental.functional.opcodes import *
from std.collections import Set
from .runtime import mil_header


struct GradStep(Movable):
    var text: String
    var args: List[Int]
    """Node id fed to each argument but the last (the seed)."""
    var grads: List[Int]
    """Parameter node id of each output after the loss."""
    var shapes: List[List[Int]]
    """Every node's [rows, cols]."""
    var grad_div: Float64
    """Divide the gradient outputs by seed * grad_div: when the loss is a
    mean, its 1/count is left to the host (count, else 1)."""

    def __init__(out self, var text: String, var args: List[Int], var grads: List[Int],
                 var shapes: List[List[Int]], grad_div: Float64):
        self.text = text^
        self.args = args^
        self.grads = grads^
        self.shapes = shapes^
        self.grad_div = grad_div


def _ty(s: List[Int]) -> String:
    return String("tensor<fp16, [1, 1, ", s[0], ", ", s[1], "]>")


struct Mil(Movable):
    """SSA lines of one MIL function body. `live(outputs)` returns only the
    lines the outputs depend on: the ANE compiler can FAIL on a dead op
    (one whose operand also feeds a live reduce did, on the M1)."""

    var lines: List[String]
    var defs: List[String]
    var n: Int

    def __init__(out self):
        self.lines = List[String]()
        self.defs = List[String]()
        self.n = 0

    def fresh(mut self) -> String:
        self.n += 1
        return String("v", self.n)

    def emit(mut self, v: String, line: String):
        self.lines.append(line)
        self.defs.append(v)

    def op(mut self, ty: String, call: String) -> String:
        """`ty v = call[name = "v"]`; returns `v`."""
        var v = self.fresh()
        self.named(ty, v, call)
        return v

    def named(mut self, ty: String, v: String, call: String):
        self.emit(v, String("        ", ty, " ", v, " = ", call, "[name = string(\"", v, "\")];\n"))

    def body(self) -> String:
        var b = String()
        for ln in self.lines:
            b += ln
        return b^

    def live(self, outputs: List[String]) -> String:
        var need = Set[String]()
        for o in outputs:
            need.add(o)
        var keep = List[Bool](length=len(self.lines), fill=False)
        for i in range(len(self.lines) - 1, -1, -1):
            if self.defs[i] not in need:
                continue
            keep[i] = True
            # operands: every `v<digits>` after the `=`
            var ln = self.lines[i]
            var rhs = String(ln[byte = ln.find(" = ") + 3 :])
            var j = 0
            var bs = rhs.as_bytes()
            while j < len(bs):
                if Int(bs[j]) == ord("v") and (j == 0 or not _ident(Int(bs[j - 1]))):
                    var k = j + 1
                    while k < len(bs) and Int(bs[k]) >= ord("0") and Int(bs[k]) <= ord("9"):
                        k += 1
                    if k > j + 1 and (k == len(bs) or not _ident(Int(bs[k]))):
                        need.add(String(rhs[byte = j : k]))
                    j = k
                else:
                    j += 1
        var b = String()
        for i in range(len(self.lines)):
            if keep[i]:
                b += self.lines[i]
        return b^

    def scalar(mut self, c: Float32) raises -> String:
        if abs(c) > 65504.0:
            raise Error(String("tape_to_mil: constant ", c, " does not fit in fp16"))
        var v = self.fresh()
        self.emit(v, String("        fp16 ", v, " = const()[name = string(\"", v, "\"), val = fp16(", c, ")];\n"))
        return v

    def ints(mut self, vals: List[Int]) -> String:
        var v = self.fresh()
        var lit = String()
        for i in range(len(vals)):
            lit += String(vals[i]) if i == 0 else String(",", vals[i])
        self.emit(v, String(
            "        tensor<int32, [", len(vals), "]> ", v, " = const()[name = string(\"", v,
            "\"), val = tensor<int32, [", len(vals), "]>([", lit, "])];\n",
        ))
        return v

    def boolean(mut self, b: Bool) -> String:
        var v = self.fresh()
        self.emit(v, String("        bool ", v, " = const()[name = string(\"", v, "\"), val = bool(",
                            "true" if b else "false", ")];\n"))
        return v

    # ── ops on [1, 1, R, K] tensors ─────────────────────────────────────
    def bin(mut self, name: String, a: String, b: String, so: List[Int]) -> String:
        return self.op(_ty(so), String(name, "(x = ", a, ", y = ", b, ")"))

    def un(mut self, name: String, a: String, so: List[Int]) -> String:
        return self.op(_ty(so), String(name, "(x = ", a, ")"))

    def scal(mut self, name: String, a: String, c: Float32, so: List[Int]) raises -> String:
        """`a <name> c`."""
        var k = self.scalar(c)
        return self.op(_ty(so), String(name, "(x = ", a, ", y = ", k, ")"))

    def rscal(mut self, name: String, c: Float32, a: String, so: List[Int]) raises -> String:
        """`c <name> a`."""
        var k = self.scalar(c)
        return self.op(_ty(so), String(name, "(x = ", k, ", y = ", a, ")"))

    def matmul(mut self, a: String, ta: Bool, b: String, tb: Bool, so: List[Int]) -> String:
        var x = self.boolean(ta)
        var y = self.boolean(tb)
        return self.op(_ty(so), String("matmul(transpose_x = ", x, ", transpose_y = ", y,
                                       ", x = ", a, ", y = ", b, ")"))

    def reduce(mut self, kind: String, a: String, axes: List[Int], so: List[Int]) -> String:
        """`reduce_<kind>` over padded axes (2 = rows, 3 = cols), keeping them."""
        var ax = self.ints(axes)
        var kd = self.boolean(True)
        return self.op(_ty(so), String("reduce_", kind, "(x = ", a, ", axes = ", ax,
                                       ", keep_dims = ", kd, ")"))

    def transpose(mut self, a: String, so: List[Int]) -> String:
        var pm = self.ints([0, 1, 3, 2])
        return self.op(_ty(so), String("transpose(perm = ", pm, ", x = ", a, ")"))

    def tile(mut self, a: String, s: List[Int], so: List[Int]) -> String:
        """Broadcast `a` (size 1 on some axes) up to `so`."""
        if s[0] == so[0] and s[1] == so[1]:
            return a
        var r = self.ints([1, 1, so[0] // s[0], so[1] // s[1]])
        return self.op(_ty(so), String("tile(x = ", a, ", reps = ", r, ")"))


def _ident(b: Int) -> Bool:
    return (b >= ord("a") and b <= ord("z")) or (b >= ord("A") and b <= ord("Z")) or (
        b >= ord("0") and b <= ord("9")) or b == ord("_")


def _act(sub: Int) raises -> String:
    if sub == EW_RELU:
        return "relu"
    elif sub == EW_TANH:
        return "tanh"
    elif sub == EW_SIGMOID:
        return "sigmoid"
    elif sub == EW_EXP:
        return "exp"
    elif sub == EW_LOG:
        return "log"
    elif sub == EW_SQRT:
        return "sqrt"
    raise Error("tape_to_mil: activation tag " + String(sub) + " not supported yet")


def _act_fwd(mut g: Mil, sub: Int, x: String, s: List[Int]) raises -> String:
    if sub == EW_LOG:
        # an explicit epsilon: MIL's default (1e-45) is not an fp16 and the
        # ANE refuses the op without one; fp16's smallest subnormal instead
        var e = g.scalar(5.9604645e-08)
        return g.op(_ty(s), String("log(x = ", x, ", epsilon = ", e, ")"))
    return g.un(_act(sub), x, s)


def _act_bwd(mut g: Mil, mutant: Bool, sub: Int, gin: String, x: String, y: String,
             s: List[Int]) raises -> String:
    """d act(x) given the incoming `gin`, the input `x` and the output `y`.
    `mutant` (gate check only): tanh's backward uses 1 + y² instead of 1 - y²."""
    if sub == EW_RELU:
        # 1 where x > 0, else 0: relu(sign(x)) (sign(0) = 0)
        var m = g.un("relu", g.un("sign", x, s), s)
        return g.bin("mul", gin, m, s)
    elif sub == EW_TANH:
        var yy = g.bin("mul", y, y, s)
        var d = g.rscal("add" if mutant else "sub", 1.0, yy, s)
        return g.bin("mul", gin, d, s)
    elif sub == EW_SIGMOID:
        var om = g.rscal("sub", 1.0, y, s)
        return g.bin("mul", gin, g.bin("mul", y, om, s), s)
    elif sub == EW_EXP:
        return g.bin("mul", gin, y, s)
    elif sub == EW_LOG:
        return g.bin("real_div", gin, x, s)
    elif sub == EW_SQRT:
        return g.bin("real_div", g.scal("mul", gin, 0.5, s), y, s)
    raise Error("tape_to_mil: no backward for activation " + String(sub))


def _acc(mut g: Mil, mut grads: List[String], rg: List[Bool], shapes: List[List[Int]],
         i: Int, val: String):
    if not rg[i]:
        return
    if grads[i] == "":
        grads[i] = val
    else:
        grads[i] = g.bin("add", grads[i], val, shapes[i])


def _div_count(mut g: Mil, a: String, n: Int, so: List[Int]) raises -> String:
    """`a / n` for an element count `n` that may exceed fp16's 65504: exact
    divisions by 256 first, then the rest."""
    var v = a
    var c = Float64(n)
    while c > 32768.0:
        v = g.scal("mul", v, 1.0 / 256.0, so)
        c /= 256.0
    return g.scal("real_div", v, Float32(c), so)


def _unb(mut g: Mil, val: String, gs: List[Int], target: List[Int]) -> String:
    """Sum `val` over every axis `target` broadcast (size 1 there)."""
    var axes = List[Int]()
    var so = gs.copy()
    for ax in range(2):
        if target[ax] == 1 and gs[ax] > 1:
            axes.append(ax + 2)
            so[ax] = 1
    if len(axes) == 0:
        return val
    return g.reduce("sum", val, axes, so)


def emit_grad_step(tape: Tape, loss: FTensor, mutant: Bool = False) raises -> GradStep:
    ref td = tape.data[]
    var n = td.n_live
    var g = Mil()
    var vals = List[String](length=n, fill=String(""))
    var pre = List[String](length=n, fill=String(""))
    var grads = List[String](length=n, fill=String(""))
    var shapes = List[List[Int]]()
    for i in range(n):
        ref nd = td.nodes[i]
        if nd.rank > 2:
            raise Error("tape_to_mil: node " + String(i) + " has rank " + String(nd.rank)
                        + " (rank-2 tapes only)")
        shapes.append([nd.rows, nd.cols])

    # ── arguments: parameters, then data inputs, then the seed ──────────
    var args = List[Int]()
    var sig = String()
    for i in range(n):
        if td.nodes[i].op == OP_LEAF:
            args.append(i)
            vals[i] = String("i", _pad3(len(args) - 1))
            sig += String(_ty(shapes[i]), " ", vals[i], ", ")
    var seed = String("i", _pad3(len(args)))
    var one1: List[Int] = [1, 1]
    sig += String(_ty(one1), " ", seed)

    # ── forward ─────────────────────────────────────────────────────────
    for i in range(n):
        ref nd = td.nodes[i]
        var op = nd.op
        if op == OP_LEAF:
            continue
        var so = shapes[i].copy()
        if op == OP_ADD or op == OP_SUB or op == OP_MUL or op == OP_DIV:
            var name = "add" if op == OP_ADD else ("sub" if op == OP_SUB else ("mul" if op == OP_MUL else "real_div"))
            vals[i] = g.bin(name, vals[nd.a], vals[nd.b], so)
        elif op == OP_MATMUL:
            vals[i] = g.matmul(vals[nd.a], False, vals[nd.b], False, so)
        elif op == OP_SCALE:
            vals[i] = g.scal("mul", vals[nd.a], Float32(nd.c), so)
        elif op == OP_SHIFT:
            vals[i] = g.scal("add", vals[nd.a], Float32(nd.c), so)
        elif op == OP_DIVC:
            vals[i] = g.scal("real_div", vals[nd.a], Float32(nd.c), so)
        elif op == OP_RSUBC:
            vals[i] = g.rscal("sub", Float32(nd.c), vals[nd.a], so)
        elif op == OP_EW:
            vals[i] = _act_fwd(g, nd.sub, vals[nd.a], so)
        elif op == OP_SUM or op == OP_MEAN:
            vals[i] = g.reduce("sum" if op == OP_SUM else "mean", vals[nd.a], [2, 3], so)
        elif op == OP_TRANSPOSE:
            vals[i] = g.transpose(vals[nd.a], so)
        elif op == OP_SUM_ROWS:
            vals[i] = g.reduce("sum", vals[nd.a], [2], so)
        elif op == OP_LINEAR:
            var z = g.matmul(vals[nd.a], False, vals[nd.b], False, so)
            z = g.bin("add", z, vals[nd.x3], so)
            pre[i] = z
            vals[i] = z if nd.sub == EW_NONE else _act_fwd(g, nd.sub, z, so)
        else:
            raise Error("tape_to_mil: op tag " + String(op) + " (node " + String(i)
                        + ") not supported yet")

    # ── backward ────────────────────────────────────────────────────────
    var L = loss.id
    if shapes[L][0] != 1 or shapes[L][1] != 1:
        raise Error("tape_to_mil: the loss must be [1, 1]")

    var rg = List[Bool]()
    for i in range(n):
        rg.append(td.nodes[i].rg)

    grads[L] = seed
    var grad_div = Float64(1)
    for i in range(L, -1, -1):
        ref nd = td.nodes[i]
        if grads[i] == "" or nd.op == OP_LEAF:
            continue
        var gv = grads[i]
        var so = shapes[i].copy()
        var op = nd.op
        if op == OP_ADD or op == OP_SUB:
            _acc(g, grads, rg, shapes, nd.a, _unb(g, gv, so, shapes[nd.a]))
            var gb = gv if op == OP_ADD else g.scal("mul", gv, -1.0, so)
            _acc(g, grads, rg, shapes, nd.b, _unb(g, gb, so, shapes[nd.b]))
        elif op == OP_MUL:
            _acc(g, grads, rg, shapes, nd.a, _unb(g, g.bin("mul", gv, vals[nd.b], so), so, shapes[nd.a]))
            _acc(g, grads, rg, shapes, nd.b, _unb(g, g.bin("mul", gv, vals[nd.a], so), so, shapes[nd.b]))
        elif op == OP_DIV:
            _acc(g, grads, rg, shapes, nd.a, _unb(g, g.bin("real_div", gv, vals[nd.b], so), so, shapes[nd.a]))
            var t = g.bin("mul", gv, vals[i], so)                 # gv * (a / b)
            t = g.bin("real_div", t, vals[nd.b], so)
            t = g.scal("mul", t, -1.0, so)
            _acc(g, grads, rg, shapes, nd.b, _unb(g, t, so, shapes[nd.b]))
        elif op == OP_MATMUL:
            _acc(g, grads, rg, shapes, nd.a, g.matmul(gv, False, vals[nd.b], True, shapes[nd.a]))
            _acc(g, grads, rg, shapes, nd.b, g.matmul(vals[nd.a], True, gv, False, shapes[nd.b]))
        elif op == OP_SCALE:
            _acc(g, grads, rg, shapes, nd.a, g.scal("mul", gv, Float32(nd.c), so))
        elif op == OP_SHIFT:
            _acc(g, grads, rg, shapes, nd.a, gv)
        elif op == OP_DIVC:
            _acc(g, grads, rg, shapes, nd.a, g.scal("real_div", gv, Float32(nd.c), so))
        elif op == OP_RSUBC:
            _acc(g, grads, rg, shapes, nd.a, g.scal("mul", gv, -1.0, so))
        elif op == OP_EW:
            _acc(g, grads, rg, shapes, nd.a, _act_bwd(g, mutant, nd.sub, gv, vals[nd.a], vals[i], so))
        elif op == OP_SUM or op == OP_MEAN:
            var gg = gv
            if op == OP_MEAN and i == L:
                grad_div = Float64(shapes[nd.a][0] * shapes[nd.a][1])
            elif op == OP_MEAN:
                gg = _div_count(g, gv, shapes[nd.a][0] * shapes[nd.a][1], so)
            _acc(g, grads, rg, shapes, nd.a, g.tile(gg, so, shapes[nd.a]))
        elif op == OP_TRANSPOSE:
            _acc(g, grads, rg, shapes, nd.a, g.transpose(gv, shapes[nd.a]))
        elif op == OP_SUM_ROWS:
            _acc(g, grads, rg, shapes, nd.a, g.tile(gv, so, shapes[nd.a]))
        elif op == OP_LINEAR:
            var gz = gv if nd.sub == EW_NONE else _act_bwd(g, mutant, nd.sub, gv, pre[i], vals[i], so)
            _acc(g, grads, rg, shapes, nd.x3, _unb(g, gz, so, shapes[nd.x3]))
            _acc(g, grads, rg, shapes, nd.b, g.matmul(vals[nd.a], True, gz, False, shapes[nd.b]))
            _acc(g, grads, rg, shapes, nd.a, g.matmul(gz, False, vals[nd.b], True, shapes[nd.a]))
        else:
            raise Error("tape_to_mil: no backward for op tag " + String(op))

    # ── outputs: the loss, then each parameter's gradient ───────────────
    var outs = List[String]()
    var pgrads = List[Int]()
    outs.append("o000")
    g.named(_ty(one1), "o000", String("identity(x = ", vals[L], ")"))
    for i in range(td.n_params):
        if grads[i] == "":
            continue
        var o = String("o", _pad3(len(outs)))
        g.named(_ty(shapes[i]), o, String("identity(x = ", grads[i], ")"))
        outs.append(o)
        pgrads.append(i)

    var text = mil_header()
    text += String("    func main<ios18>(", sig, ") {\n")
    text += g.live(outs)
    var ret = String()
    for k in range(len(outs)):
        ret += outs[k] if k == 0 else String(", ", outs[k])
    text += String("    } -> (", ret, ");\n}\n")
    return GradStep(text^, args^, pgrads^, shapes^, grad_div)


def _pad3(k: Int) -> String:
    var s = String(k)
    while s.byte_length() < 3:
        s = "0" + s
    return s^
