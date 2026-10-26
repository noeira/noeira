"""Which MIL ops does this ANE compile? One tiny program per op, each built
with the emitter's own `Mil` writer (`emit.mojo`), on `[1, 1, 8, 32]` fp16
tensors. `ANECCompile() FAILED` names no op, so this is how to find one.

    pixi run -e apple mojo run -I . benchmarks/ane/probe_ops.mojo
"""

from std.math import exp, log, sqrt, tanh
from noeira.experimental.ane.runtime import ANE, Program, mil_header
from noeira.experimental.ane.emit import Mil, _ty


def try_op(a: ANE, name: String, mut g: Mil, res: String, so: List[Int], s: List[Int] = [8, 32]):
    var text = mil_header()
    text += String("    func main<ios18>(", _ty(s), " i000, ", _ty(s), " i001) {\n")
    text += g.body()
    text += String("        ", _ty(so), " o000 = add(x = ", res, ", y = ", res, ")[name = string(\"o000\")];\n")
    text += "    } -> (o000);\n}\n"
    var pool = a.o.pool_push()
    try:
        var p = Program(a, text)
        p.run(a)
        print("  ", name, ": OK")
        p.free(a)
    except e:
        print("  ", name, ": FAIL", e)
    a.o.pool_pop(pool)


def main() raises:
    var a = ANE()
    var s: List[Int] = [8, 32]
    var r1: List[Int] = [1, 32]
    var c1: List[Int] = [8, 1]
    var one: List[Int] = [1, 1]
    var sq: List[Int] = [32, 32]
    var e8: List[Int] = [8, 8]
    for name in ["identity", "sign", "relu", "tanh", "sigmoid", "exp", "log", "sqrt"]:
        var g = Mil()
        var v = g.un(name, "i000", s)
        try_op(a, name, g, v, s)
    for name in ["add", "sub", "mul", "real_div"]:
        var g = Mil()
        var v = g.bin(name, "i000", "i001", s)
        try_op(a, name, g, v, s)
    var g = Mil()
    var v = g.scal("mul", "i000", 0.5, s)
    try_op(a, "mul by fp16 scalar const", g, v, s)
    g = Mil()
    v = g.rscal("sub", 1.0, "i000", s)
    try_op(a, "scalar const - x", g, v, s)
    g = Mil()
    v = g.reduce("sum", "i000", [2], r1)
    try_op(a, "reduce_sum rows", g, v, r1)
    g = Mil()
    v = g.reduce("sum", "i000", [3], c1)
    try_op(a, "reduce_sum cols", g, v, c1)
    g = Mil()
    v = g.reduce("sum", "i000", [2, 3], one)
    try_op(a, "reduce_sum all", g, v, one)
    g = Mil()
    v = g.reduce("mean", "i000", [2, 3], one)
    try_op(a, "reduce_mean all", g, v, one)
    g = Mil()
    v = g.tile(g.reduce("sum", "i000", [2], r1), r1, s)
    try_op(a, "tile rows", g, v, s)
    g = Mil()
    v = g.tile(g.reduce("sum", "i000", [2, 3], one), one, s)
    try_op(a, "tile all", g, v, s)
    g = Mil()
    v = g.bin("add", "i000", g.reduce("sum", "i001", [2], r1), s)
    try_op(a, "add broadcast [1,K]", g, v, s)
    g = Mil()
    v = g.matmul("i000", True, "i001", False, sq)
    try_op(a, "matmul x^T y", g, v, sq)
    g = Mil()
    v = g.matmul("i000", False, "i001", True, e8)
    try_op(a, "matmul x y^T", g, v, e8)
    g = Mil()
    v = g.transpose("i000", [32, 8])
    try_op(a, "transpose", g, v, [32, 8])

    # reduce over rows: which shapes?
    for rk in [(8, 32), (64, 32), (64, 64), (64, 6), (64, 17), (8, 6), (8, 1), (64, 1), (256, 64), (256, 1)]:
        var sh: List[Int] = [rk[0], rk[1]]
        var o1: List[Int] = [1, rk[1]]
        g = Mil()
        v = g.reduce("sum", "i000", [2], o1)
        try_op(a, String("reduce_sum rows ", rk[0], "x", rk[1]), g, v, o1, sh)
        var ot: List[Int] = [rk[1], rk[0]]
        var oc: List[Int] = [rk[1], 1]
        g = Mil()
        v = g.reduce("sum", g.transpose("i000", ot), [3], oc)
        try_op(a, String("  transpose + reduce cols ", rk[0], "x", rk[1]), g, v, oc, sh)

    # accuracy of the ANE's elementwise functions, in fp16 ulps (2^-11 relative)
    print("accuracy: max error over 256 points, in units of 2^-11 * |f(x)|")
    for cs in [("exp", -4.0, 4.0), ("log", 0.1, 10.0), ("sqrt", 0.01, 10.0), ("sigmoid", -8.0, 8.0),
                 ("tanh", -4.0, 4.0), ("recip", 0.25, 4.0)]:
        var name = String(cs[0])
        var lo = Float64(cs[1])
        var hi = Float64(cs[2])
        g = Mil()
        if name == "log":
            var ep = g.scalar(5.9604645e-08)
            v = g.op("tensor<fp16, [1, 1, 8, 32]>", String("log(x = i000, epsilon = ", ep, ")"))
        elif name == "recip":
            v = g.bin("real_div", "i001", "i000", s)
        else:
            v = g.un(name, "i000", s)
        var text = mil_header()
        text += String("    func main<ios18>(", _ty(s), " i000, ", _ty(s), " i001) {\n")
        text += g.body()
        text += String("        ", _ty(s), " o000 = identity(x = ", v, ")[name = string(\"o000\")];\n")
        text += "    } -> (o000);\n}\n"
        var pool = a.o.pool_push()
        var p = Program(a, text)
        var x = List[Float32](length=256, fill=0)
        var ones = List[Float32](length=256, fill=1)
        for i in range(256):
            x[i] = Float32(lo + (hi - lo) * Float64(i) / 255.0).cast[DType.float16]().cast[DType.float32]()
        p.write(a, 0, Pointer(to=x[0]))
        p.write(a, 1, Pointer(to=ones[0]))
        p.run(a)
        var y = List[Float32](length=256, fill=0)
        p.read(a, 0, Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(y.unsafe_ptr())))
        var worst = Float64(0)
        var at = Float64(0)
        for i in range(256):
            var xv = Float64(x[i])
            var want: Float64
            if name == "exp":
                want = exp(xv)
            elif name == "log":
                want = log(xv)
            elif name == "sqrt":
                want = sqrt(xv)
            elif name == "sigmoid":
                want = 1.0 / (1.0 + exp(-xv))
            elif name == "tanh":
                want = tanh(xv)
            else:
                want = 1.0 / xv
            var e = abs(Float64(y[i]) - want) / max(abs(want), 1e-3) / 2.0 ** -11
            if e > worst:
                worst = e
                at = xv
        print("  ", name, ": worst", worst, "ulps at x =", at)
        p.free(a)
        _ = y^
        _ = x^
        _ = ones^
        a.o.pool_pop(pool)
