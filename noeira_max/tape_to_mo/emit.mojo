"""Functional tape -> one MO graph for MAX: the recorded forward, its
backward (each op's VJP, written here as MO ops), and Adam updating the
parameters, its moments and its bias-correction powers IN PLACE.

    var step = emit_train_step(tape, loss, lr=1e-3)   # after ONE eager forward
    # step.text: a `mo.graph` MAX's C API compiles (M_setModelPath on a file)
    # arguments, in order: p<i> (buffer), m<i>, v<i> (buffers), pow1, pow2
    # (buffers, [1, 1], start at 1.0), then x<id> for each data input
    # result: "loss" ([1, 1]), the loss BEFORE the update

Rank-2 graphs only (this spike's first phase): the ops an MLP and its loss
record — add/sub/mul/div with broadcasting, matmul, scalar ops, the
activations below, sum/mean, transpose, sum_rows, and `F.linear` with or
without an activation. Anything else raises, naming the op.
"""

from noeira.nn.constants import DT
from noeira.experimental.functional import Tape, FTensor
from noeira.experimental.functional.opcodes import *

from .mo_text import MoGraph, shape_ty, buffer_ty


struct Step(Movable):
    var text: String
    var params: List[Int]
    """Parameter node ids; argument `p<k>` is `params[k]`."""
    var inputs: List[Int]
    """Data-input node ids; argument `x<id>`."""
    var shapes: List[List[Int]]
    """Every node's [rows, cols]."""

    def __init__(out self, var text: String, var params: List[Int], var inputs: List[Int],
                 var shapes: List[List[Int]]):
        self.text = text^
        self.params = params^
        self.inputs = inputs^
        self.shapes = shapes^


def _ew_name(sub: Int) raises -> String:
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
    raise Error("tape_to_mo: activation tag " + String(sub) + " not supported yet")


comptime SCALAR_T = "!mo.tensor<[], f32>"


def scal(mut g: MoGraph, op: String, a: String, shape: List[Int], c: Float32) -> String:
    """`a <op> c` for a scalar constant."""
    var k = g.const(c)
    return g.binary(op, a, shape_ty(shape), k, SCALAR_T, shape_ty(shape))


def rscal(mut g: MoGraph, op: String, c: Float32, a: String, shape: List[Int]) -> String:
    """`c <op> a`."""
    var k = g.const(c)
    return g.binary(op, k, SCALAR_T, a, shape_ty(shape), shape_ty(shape))


def bin(mut g: MoGraph, op: String, a: String, sa: List[Int], b: String, sb: List[Int], so: List[Int]) -> String:
    return g.binary(op, a, shape_ty(sa), b, shape_ty(sb), shape_ty(so))


def unb(mut g: MoGraph, val: String, gs: List[Int], target: List[Int]) -> String:
    """Sum `val` over every axis `target` broadcast (size 1 there)."""
    var cur = val
    var s = gs.copy()
    for ax in range(2):
        if target[ax] == 1 and s[ax] > 1:
            var o = s.copy()
            o[ax] = 1
            cur = g.reduce("add", cur, shape_ty(s), ax, shape_ty(o))
            s = o^
    return cur


def acc(mut g: MoGraph, mut grads: List[String], shapes: List[List[Int]], i: Int, val: String):
    if grads[i] == "":
        grads[i] = val
    else:
        var prev = grads[i]
        grads[i] = bin(g, "add", prev, shapes[i], val, shapes[i], shapes[i])


def act_fwd(mut g: MoGraph, sub: Int, x: String, s: List[Int]) raises -> String:
    return g.unary(_ew_name(sub), x, shape_ty(s), shape_ty(s))


def act_bwd(mut g: MoGraph, mutant: Bool, sub: Int, gin: String, x: String, y: String,
            s: List[Int]) raises -> String:
    """d act(x) given the incoming `gin`, the input `x` and the output `y`.
    `mutant` (gate check only): tanh's backward uses 1 + y² instead of 1 - y²."""
    if sub == EW_RELU:
        var zero = g.const(0.0)
        var pos = g.binary("greater", x, shape_ty(s), zero, SCALAR_T, shape_ty(s, "bool"))
        var zero2 = g.const(0.0)
        return g.select(pos, shape_ty(s, "bool"), gin, shape_ty(s), zero2, SCALAR_T, shape_ty(s))
    elif sub == EW_TANH:
        var yy = bin(g, "mul", y, s, y, s, s)
        var d = rscal(g, "add" if mutant else "sub", 1.0, yy, s)
        return bin(g, "mul", gin, s, d, s, s)
    elif sub == EW_SIGMOID:
        var om = rscal(g, "sub", 1.0, y, s)
        var d = bin(g, "mul", y, s, om, s, s)
        return bin(g, "mul", gin, s, d, s, s)
    elif sub == EW_EXP:
        return bin(g, "mul", gin, s, y, s, s)
    elif sub == EW_LOG:
        return bin(g, "div", gin, s, x, s, s)
    elif sub == EW_SQRT:
        var h = scal(g, "mul", gin, s, 0.5)
        return bin(g, "div", h, s, y, s, s)
    raise Error("tape_to_mo: no backward for activation " + String(sub))


def emit_train_step(
    tape: Tape, loss: FTensor, lr: Float32, beta1: Float32 = 0.9,
    beta2: Float32 = 0.999, eps: Float32 = 1e-8, mutant: Bool = False,
    device: String = "",
    extra: List[Int] = [],
    update: Bool = True,
    grads_out: Bool = False,
    adam_out: Bool = False,
) raises -> Step:
    """`grads_out` (timing stages): also outputs each parameter's gradient
    as "g<k>". `device`: "" = the host; "gpu:0" puts every tensor and buffer (and the
    scalar constants) on the GPU, as the Python API prints it — transpose's
    si64 permutation constant stays on the host."""
    ref td = tape.data[]
    var n = td.n_live
    var g = MoGraph("tape_step")
    var vals = List[String](length=n, fill=String(""))
    var pre = List[String](length=n, fill=String(""))
    var grads = List[String](length=n, fill=String(""))
    var shapes = List[List[Int]]()
    for i in range(n):
        ref nd = td.nodes[i]
        if nd.rank > 2:
            raise Error("tape_to_mo: node " + String(i) + " has rank " + String(nd.rank)
                        + " (rank-2 graphs only in this phase)")
        shapes.append([nd.rows, nd.cols])

    # ── arguments: parameters, moments, powers, then data inputs ────────
    var params = List[Int]()
    var inputs = List[Int]()
    var pbuf = List[String]()
    var mbuf = List[String]()
    var vbuf = List[String]()
    for i in range(td.n_params):
        params.append(i)
        pbuf.append(g.arg(String("p", i), buffer_ty(shapes[i])))
    for i in range(td.n_params):
        mbuf.append(g.arg(String("m", i), buffer_ty(shapes[i])))
    for i in range(td.n_params):
        vbuf.append(g.arg(String("v", i), buffer_ty(shapes[i])))
    var one1: List[Int] = [1, 1]
    var pow1 = g.arg("pow1", buffer_ty(one1))
    var pow2 = g.arg("pow2", buffer_ty(one1))
    for i in range(td.n_params, n):
        if td.nodes[i].op == OP_LEAF:
            inputs.append(i)
            vals[i] = g.arg(String("x", i), shape_ty(shapes[i]))

    # every buffer is loaded before anything is stored (one chain, in order)
    var mval = List[String]()
    var vval = List[String]()
    for k in range(len(params)):
        vals[params[k]] = g.load(pbuf[k], shapes[params[k]])
        mval.append(g.load(mbuf[k], shapes[params[k]]))
        vval.append(g.load(vbuf[k], shapes[params[k]]))
    var p1 = g.load(pow1, one1)
    var p2 = g.load(pow2, one1)

    # ── forward ─────────────────────────────────────────────────────────
    for i in range(td.n_params, n):
        ref nd = td.nodes[i]
        var op = nd.op
        if op == OP_LEAF:
            continue
        var so = shapes[i].copy()
        if op == OP_ADD or op == OP_SUB or op == OP_MUL or op == OP_DIV:
            var name = "add" if op == OP_ADD else ("sub" if op == OP_SUB else ("mul" if op == OP_MUL else "div"))
            vals[i] = bin(g, name, vals[nd.a], shapes[nd.a], vals[nd.b], shapes[nd.b], so)
        elif op == OP_MATMUL:
            vals[i] = bin(g, "matmul", vals[nd.a], shapes[nd.a], vals[nd.b], shapes[nd.b], so)
        elif op == OP_SCALE:
            vals[i] = scal(g, "mul", vals[nd.a], so, Float32(nd.c))
        elif op == OP_SHIFT:
            vals[i] = scal(g, "add", vals[nd.a], so, Float32(nd.c))
        elif op == OP_DIVC:
            vals[i] = scal(g, "div", vals[nd.a], so, Float32(nd.c))
        elif op == OP_RSUBC:
            vals[i] = rscal(g, "sub", Float32(nd.c), vals[nd.a], so)
        elif op == OP_EW:
            vals[i] = act_fwd(g, nd.sub, vals[nd.a], so)
        elif op == OP_SUM or op == OP_MEAN:
            var kind = "add" if op == OP_SUM else "mean"
            var sa = shapes[nd.a].copy()
            var mid: List[Int] = [sa[0], 1]
            var r = g.reduce(kind, vals[nd.a], shape_ty(sa), 1, shape_ty(mid))
            vals[i] = g.reduce(kind, r, shape_ty(mid), 0, shape_ty(so))
        elif op == OP_TRANSPOSE:
            vals[i] = g.transpose2(vals[nd.a], shape_ty(shapes[nd.a]), shape_ty(so))
        elif op == OP_SUM_ROWS:
            vals[i] = g.reduce("add", vals[nd.a], shape_ty(shapes[nd.a]), 0, shape_ty(so))
        elif op == OP_LINEAR:
            var z = bin(g, "matmul", vals[nd.a], shapes[nd.a], vals[nd.b], shapes[nd.b], so)
            z = bin(g, "add", z, so, vals[nd.x3], shapes[nd.x3], so)
            pre[i] = z
            vals[i] = z if nd.sub == EW_NONE else act_fwd(g, nd.sub, z, so)
        else:
            raise Error("tape_to_mo: op tag " + String(op) + " (node " + String(i)
                        + ") not supported yet")

    # ── backward ────────────────────────────────────────────────────────
    var L = loss.id
    if shapes[L][0] != 1 or shapes[L][1] != 1:
        raise Error("tape_to_mo: the loss must be [1, 1]")
    var seed = g.const(1.0)
    grads[L] = g.broadcast_to(seed, "!mo.tensor<[], f32>", shapes[L])
    for i in range(L, td.n_params - 1, -1):
        ref nd = td.nodes[i]
        if grads[i] == "" or nd.op == OP_LEAF:
            continue
        var gv = grads[i]
        var so = shapes[i].copy()
        var op = nd.op
        if op == OP_ADD or op == OP_SUB:
            acc(g, grads, shapes, nd.a, unb(g, gv, so, shapes[nd.a]))
            var gb = gv if op == OP_ADD else scal(g, "mul", gv, so, -1.0)
            acc(g, grads, shapes, nd.b, unb(g, gb, so, shapes[nd.b]))
        elif op == OP_MUL:
            var ga = bin(g, "mul", gv, so, vals[nd.b], shapes[nd.b], so)
            acc(g, grads, shapes, nd.a, unb(g, ga, so, shapes[nd.a]))
            var gb = bin(g, "mul", gv, so, vals[nd.a], shapes[nd.a], so)
            acc(g, grads, shapes, nd.b, unb(g, gb, so, shapes[nd.b]))
        elif op == OP_DIV:
            var ga = bin(g, "div", gv, so, vals[nd.b], shapes[nd.b], so)
            acc(g, grads, shapes, nd.a, unb(g, ga, so, shapes[nd.a]))
            var t = bin(g, "mul", gv, so, vals[i], so, so)          # gv * (a / b)
            t = bin(g, "div", t, so, vals[nd.b], shapes[nd.b], so)
            t = scal(g, "mul", t, so, -1.0)
            acc(g, grads, shapes, nd.b, unb(g, t, so, shapes[nd.b]))
        elif op == OP_MATMUL:
            var sa = shapes[nd.a].copy()
            var sb = shapes[nd.b].copy()
            var sbt: List[Int] = [sb[1], sb[0]]
            var sat: List[Int] = [sa[1], sa[0]]
            var bt = g.transpose2(vals[nd.b], shape_ty(sb), shape_ty(sbt))
            acc(g, grads, shapes, nd.a, bin(g, "matmul", gv, so, bt, sbt, sa))
            var at = g.transpose2(vals[nd.a], shape_ty(sa), shape_ty(sat))
            acc(g, grads, shapes, nd.b, bin(g, "matmul", at, sat, gv, so, sb))
        elif op == OP_SCALE:
            acc(g, grads, shapes, nd.a, scal(g, "mul", gv, so, Float32(nd.c)))
        elif op == OP_SHIFT:
            acc(g, grads, shapes, nd.a, gv)
        elif op == OP_DIVC:
            acc(g, grads, shapes, nd.a, scal(g, "div", gv, so, Float32(nd.c)))
        elif op == OP_RSUBC:
            acc(g, grads, shapes, nd.a, scal(g, "mul", gv, so, -1.0))
        elif op == OP_EW:
            acc(g, grads, shapes, nd.a, act_bwd(g, mutant, nd.sub, gv, vals[nd.a], vals[i], so))
        elif op == OP_SUM or op == OP_MEAN:
            var gg = gv
            if op == OP_MEAN:
                var cnt = shapes[nd.a][0] * shapes[nd.a][1]
                gg = scal(g, "div", gv, so, Float32(cnt))
            acc(g, grads, shapes, nd.a, g.broadcast_to(gg, shape_ty(so), shapes[nd.a]))
        elif op == OP_TRANSPOSE:
            acc(g, grads, shapes, nd.a, g.transpose2(gv, shape_ty(so), shape_ty(shapes[nd.a])))
        elif op == OP_SUM_ROWS:
            acc(g, grads, shapes, nd.a, g.broadcast_to(gv, shape_ty(so), shapes[nd.a]))
        elif op == OP_LINEAR:
            var gz = gv if nd.sub == EW_NONE else act_bwd(g, mutant, nd.sub, gv, pre[i], vals[i], so)
            acc(g, grads, shapes, nd.x3, unb(g, gz, so, shapes[nd.x3]))
            var sa = shapes[nd.a].copy()
            var sb = shapes[nd.b].copy()
            var sat: List[Int] = [sa[1], sa[0]]
            var sbt: List[Int] = [sb[1], sb[0]]
            var at = g.transpose2(vals[nd.a], shape_ty(sa), shape_ty(sat))
            acc(g, grads, shapes, nd.b, bin(g, "matmul", at, sat, gz, so, sb))
            if td.nodes[nd.a].rg:
                var bt = g.transpose2(vals[nd.b], shape_ty(sb), shape_ty(sbt))
                acc(g, grads, shapes, nd.a, bin(g, "matmul", gz, so, bt, sbt, sa))
        else:
            raise Error("tape_to_mo: no backward for op tag " + String(op))

    # ── Adam, in place (nn's `_adam_update_kernel_rt`, op for op) ───────
    var p1n = scal(g, "mul", p1, one1, beta1)
    var p2n = scal(g, "mul", p2, one1, beta2)
    var bc1 = rscal(g, "sub", 1.0, p1n, one1)
    var bc2 = rscal(g, "sub", 1.0, p2n, one1)
    var updates = List[String]()
    var m_new = List[String]()
    var v_new = List[String]()
    for k in range(len(params)):
        var pi = params[k]
        var s = shapes[pi].copy()
        var gr = grads[pi]
        if gr == "":
            raise Error("tape_to_mo: parameter " + String(pi) + " gets no gradient")
        var mt = bin(g, "add", scal(g, "mul", mval[k], s, beta1), s, rscal(g, "mul", 1.0 - beta1, gr, s), s, s)
        var gg = bin(g, "mul", rscal(g, "mul", 1.0 - beta2, gr, s), s, gr, s, s)   # (1-b2)*g*g
        var vt = bin(g, "add", scal(g, "mul", vval[k], s, beta2), s, gg, s, s)
        var mh = bin(g, "div", mt, s, bc1, one1, s)
        var vh = bin(g, "div", vt, s, bc2, one1, s)
        var num = rscal(g, "mul", lr, mh, s)
        var den = scal(g, "add", g.unary("sqrt", vh, shape_ty(s), shape_ty(s)), s, eps)
        var upd = bin(g, "div", num, s, den, s, s)
        updates.append(bin(g, "sub", vals[pi], s, upd, s, s))
        m_new.append(mt)
        v_new.append(vt)
    if update:   # `update=False`: debugging only, the step stores nothing
        for k in range(len(params)):
            var s = shapes[params[k]].copy()
            g.store(pbuf[k], updates[k], s)
            g.store(mbuf[k], m_new[k], s)
            g.store(vbuf[k], v_new[k], s)
        g.store(pow1, p1n, one1)
        g.store(pow2, p2n, one1)
    g.output(vals[L], shape_ty(shapes[L]), "loss")
    if adam_out:   # timing: Adam computed, the new values OUTPUT instead of stored
        for k in range(len(params)):
            var s = shapes[params[k]].copy()
            g.output(updates[k], shape_ty(s), String("P", k))
            g.output(m_new[k], shape_ty(s), String("M", k))
            g.output(v_new[k], shape_ty(s), String("V", k))
        g.output(p1n, shape_ty(one1), "pow1n")
        g.output(p2n, shape_ty(one1), "pow2n")
    if grads_out:
        for k in range(len(params)):
            g.output(grads[params[k]], shape_ty(shapes[params[k]]), String("g", k))
    for i in range(len(extra)):   # debugging: node values, before the update
        g.output(vals[extra[i]], shape_ty(shapes[extra[i]]), String("n", extra[i]))
    var text = g.text()
    if device != "":
        text = text.replace(", f32>", ", f32, " + device + ">").replace(", bool>", ", bool, " + device + ">")
    return Step(text^, params^, inputs^, shapes^)
