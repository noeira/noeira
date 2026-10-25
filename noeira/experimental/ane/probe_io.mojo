"""Step 2 prerequisites: which input/output shapes does the ANE take?

Each case compiles a tiny program, writes known fp16 inputs through the
layout the loaded model reports (`Program.write`: rows padded to 64 bytes),
runs it, and compares every output element with the CPU. Several inputs and
outputs, odd sizes, `[1, C, 1, S]` and `[1, 1, R, K]` tensors, and a linear
layer with a broadcast bias.

    pixi run -e apple mojo run -I . noeira/experimental/ane/probe_io.mojo
"""

from noeira.experimental.ane.runtime import ANE, Program, mil_header


def t4(d0: Int, d1: Int, d2: Int, d3: Int) -> String:
    return String("tensor<fp16, [", d0, ", ", d1, ", ", d2, ", ", d3, "]>")


def val(i: Int, k: Int) -> Float32:
    """A deterministic small value, exact in fp16."""
    return Float32((i * 7 + k * 13) % 29 - 14) / 8.0


struct Case(Movable):
    var name: String
    var mil: String
    var in_n: List[Int]  # elements per input
    var out_n: List[Int]

    def __init__(out self, name: String, mil: String, var in_n: List[Int], var out_n: List[Int]):
        self.name = name
        self.mil = mil
        self.in_n = in_n^
        self.out_n = out_n^


def run_case(a: ANE, c: Case, want: def(List[List[Float32]], Int, Int) thin -> Float32) -> Bool:
    var pool = a.o.pool_push()
    var ok = True
    try:
        var p = Program(a, c.mil)
        var xs = List[List[Float32]]()
        for k in range(len(c.in_n)):
            var x = List[Float32](length=c.in_n[k], fill=0)
            for i in range(c.in_n[k]):
                x[i] = val(i, k)
            p.write(a, k, Pointer(to=x[0]))
            xs.append(x^)
        p.run(a)
        var bad = 0
        var first = -1
        for k in range(len(c.out_n)):
            var y = List[Float32](length=c.out_n[k], fill=0)
            p.read(a, k, Pointer[Float32, MutAnyOrigin](unsafe_from_address=Int(y.unsafe_ptr())))
            for i in range(c.out_n[k]):
                var w = want(xs, k, i)
                if abs(y[i] - w) > 1e-2 + 2e-3 * abs(w):
                    if first < 0:
                        first = k * 1000000 + i
                    bad += 1
            _ = y^
        ok = bad == 0
        print("  ", c.name, ": compile", p.compile_ms, "ms,", "OK" if ok else String("WRONG ", bad, " elements, first out/idx ", first))
        p.free(a)
    except e:
        print("  ", c.name, ": FAIL", e)
        ok = False
    a.o.pool_pop(pool)
    return ok


def main() raises:
    var a = ANE()
    var results = List[Bool]()

    # (a) two inputs, two outputs
    var m = mil_header()
    m += String("    func main<ios18>(", t4(1, 64, 1, 64), " a, ", t4(1, 64, 1, 64), " b) {\n")
    m += String("        ", t4(1, 64, 1, 64), " y1 = add(x = a, y = b)[name = string(\"y1\")];\n")
    m += String("        ", t4(1, 64, 1, 64), " y2 = mul(x = a, y = b)[name = string(\"y2\")];\n")
    m += "    } -> (y1, y2);\n}\n"
    def w_a(xs: List[List[Float32]], k: Int, i: Int) -> Float32:
        return xs[0][i] + xs[1][i] if k == 0 else xs[0][i] * xs[1][i]
    results.append(run_case(a, Case("2 in, 2 out [1,64,1,64]", m, [4096, 4096], [4096, 4096]), w_a))

    # (b) odd sizes, one input
    def w_b(xs: List[List[Float32]], k: Int, i: Int) -> Float32:
        return xs[0][i] + xs[0][i]
    for cs in [(17, 6), (6, 17), (3, 64), (64, 3), (1, 1), (1, 32), (32, 1), (5, 40)]:
        var C = cs[0]
        var S = cs[1]
        m = mil_header()
        m += String("    func main<ios18>(", t4(1, C, 1, S), " x) {\n")
        m += String("        ", t4(1, C, 1, S), " y = add(x = x, y = x)[name = string(\"y\")];\n")
        m += "    } -> (y);\n}\n"
        results.append(run_case(a, Case(String("x+x [1,", C, ",1,", S, "]"), m, [C * S], [C * S]), w_b))

    # (c) a [1,1,R,K] input (matrix form) straight in, no reshape
    for cs in [(32, 64), (17, 6)]:
        var R = cs[0]
        var K = cs[1]
        m = mil_header()
        m += String("    func main<ios18>(", t4(1, 1, R, K), " x) {\n")
        m += String("        ", t4(1, 1, R, K), " y = add(x = x, y = x)[name = string(\"y\")];\n")
        m += "    } -> (y);\n}\n"
        results.append(run_case(a, Case(String("x+x [1,1,", R, ",", K, "]"), m, [R * K], [R * K]), w_b))

    # (d) a linear layer in matrix form: x [1,1,B,I] @ W [1,1,I,O] + b [1,1,1,O]
    for cs in [(64, 32, 48), (17, 5, 6), (256, 1, 64), (256, 64, 1)]:
        var B = cs[0]
        var I = cs[1]
        var O = cs[2]
        m = mil_header()
        m += String("    func main<ios18>(", t4(1, 1, B, I), " i0, ", t4(1, 1, I, O), " i1, ", t4(1, 1, 1, O), " i2) {\n")
        m += "        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n"
        m += String("        ", t4(1, 1, B, O), " z = matmul(transpose_x = bF, transpose_y = bF, x = i0, y = i1)[name = string(\"z\")];\n")
        m += String("        ", t4(1, 1, B, O), " y = add(x = z, y = i2)[name = string(\"y\")];\n")
        m += "    } -> (y);\n}\n"
        # want(k=0, i = r*O + c) = sum_d x[r, d] w[d, c] + b[c]; captured shape via globals is
        # not possible in a def literal, so the check uses the sizes stored in xs.
        def w_d(xs: List[List[Float32]], k: Int, i: Int) -> Float32:
            var O = len(xs[2])
            var I = len(xs[1]) // O
            var r = i // O
            var c = i % O
            var s = xs[2][c]
            for d in range(I):
                s += xs[0][r * I + d] * xs[1][d * O + c]
            return s
        results.append(run_case(a, Case(String("linear B", B, " I", I, " O", O), m, [B * I, I * O, O], [B * O]), w_d))

    var all_ok = True
    for r in results:
        all_ok = all_ok and r
    print("ALL OK" if all_ok else "SOME CASES FAILED (see above)")
