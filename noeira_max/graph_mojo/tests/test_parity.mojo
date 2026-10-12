"""Parity of the generated Mojo graph builder with `max.graph`.

Each case builds a graph in Mojo through `max_graph_gen` (the generated
ops, and the hand-written layer over them), compiles it to a MEF, and runs
it through `maxrt`. `reference_graphs.py` builds the same graph with
`max.graph` in Python. Both MEFs run on the same inputs, and every output
must match bit for bit.

`generated_only` uses three ops that no hand-written code mentions (`top_k`,
`mo_cumsum`, `mo_erf`): they exist in Mojo because the generator found them
in MAX's stubs.

    noeira_max/graph_mojo/run.sh
"""

from std.math import sin
from std.python import Python
from std.sys import argv

from max_graph_gen import (
    Dim,
    Graph,
    PythonBackend,
    TensorType,
    Value,
    exp,
    ops,
    reduce_max,
    reduce_mean,
    reduce_sum,
    relu,
    reshape,
    softmax,
    sqrt,
    tanh,
    transpose,
)
from maxrt import HostBuffer, Model, Runtime, from_m_dtype


def f32(var shape: List[Dim]) -> TensorType:
    return TensorType(DType.float32, shape^)


def fill(n: Int, seed: Float64) -> HostBuffer:
    var b = HostBuffer(4 * n)
    var p = b.data[Float32]()
    for i in range(n):
        p[unsafe_offset=i] = Float32(sin(Float64(i) * 0.731 + seed) * 2.0)
    return b^


def nbytes(dtype: DType, n: Int) raises -> Int:
    if dtype == DType.float32 or dtype == DType.int32:
        return 4 * n
    if dtype == DType.float64 or dtype == DType.int64:
        return 8 * n
    raise Error("no size for " + String(dtype))


def compare(
    rt: Runtime, dir: String, case_name: String, shapes: List[List[Int]], mojo: Model, n_outputs: Int
) raises -> String:
    """Runs `case_name`'s Mojo-built model and its Python-built reference on the
    same inputs; raises unless every output matches bit for bit."""
    var refs = Python.import_module("noeira_max.graph_mojo.tests.reference_graphs")
    var path = dir + "/" + case_name + "_python.mef"
    var py_compile_s = Float64(py=refs.build(case_name, path))
    var python = rt.load(path)
    var mojo_inputs = rt.tensor_map()
    var python_inputs = rt.tensor_map()
    for i in range(len(shapes)):
        var n = 1
        for d in shapes[i]:
            n *= d
        var buf = fill(n, Float64(i))
        mojo_inputs.borrow("input" + String(i), HostBuffer(copy_from=buf.address(), nbytes=4 * n), DType.float32, shapes[i])
        python_inputs.borrow("input" + String(i), buf^, DType.float32, shapes[i])
    var mo = mojo.execute(mojo_inputs)
    var po = python.execute(python_inputs)
    var checked = 0
    for k in range(n_outputs):
        var a = mo.tensor("output" + String(k))
        var b = po.tensor("output" + String(k))
        var n = a.num_elements()
        if n != b.num_elements() or a.dtype_code() != b.dtype_code():
            raise Error(case_name + " output " + String(k) + ": shape or dtype differs")
        var size = nbytes(from_m_dtype(a.dtype_code()), n)
        var pa = a.data[UInt8]()
        var pb = b.data[UInt8]()
        for j in range(size):
            if pa[unsafe_offset=j] != pb[unsafe_offset=j]:
                raise Error(case_name + " output " + String(k) + ": byte " + String(j) + " differs")
        checked += n
    return String(n_outputs) + " outputs, " + String(checked) + " values bit-identical (Python compile " + String(Int(py_compile_s * 10.0) // 10) + " s)"


def case_elementwise(rt: Runtime, dir: String) raises -> String:
    var g = Graph("elementwise_mojo", [f32([4, 8]), f32([4, 8])])
    var ins = g.inputs()
    ref x = ins[0]
    ref y = ins[1]
    g.output([(x + y) * x - y, relu(x), tanh(x), exp(x), sqrt(exp(x)), x / exp(y)])
    var m = g.compile(rt, dir + "/elementwise_mojo.mef")
    return compare(rt, dir, "elementwise", [[4, 8], [4, 8]], m, 6)


def case_matmul(rt: Runtime, dir: String) raises -> String:
    var g = Graph("matmul_mojo", [f32([4, 8]), f32([8, 3])])
    var ins = g.inputs()
    g.output([ins[0] @ ins[1]])
    var m = g.compile(rt, dir + "/matmul_mojo.mef")
    return compare(rt, dir, "matmul", [[4, 8], [8, 3]], m, 1)


def case_reshape_transpose(rt: Runtime, dir: String) raises -> String:
    var g = Graph("reshape_transpose_mojo", [f32([2, 3, 4])])
    var x = g.inputs()[0].copy()
    g.output([transpose(reshape(x, [Dim(6), Dim(4)]), 0, 1)])
    var m = g.compile(rt, dir + "/reshape_transpose_mojo.mef")
    return compare(rt, dir, "reshape_transpose", [[2, 3, 4]], m, 1)


def case_reductions(rt: Runtime, dir: String) raises -> String:
    var g = Graph("reductions_mojo", [f32([4, 8])])
    var x = g.inputs()[0].copy()
    g.output([reduce_sum(x, -1), reduce_max(x, 0), reduce_mean(x, 1)])
    var m = g.compile(rt, dir + "/reductions_mojo.mef")
    return compare(rt, dir, "reductions", [[4, 8]], m, 3)


def case_softmax(rt: Runtime, dir: String) raises -> String:
    var g = Graph("softmax_mojo", [f32([4, 8])])
    var x = g.inputs()[0].copy()
    g.output([softmax(x, -1)])
    var m = g.compile(rt, dir + "/softmax_mojo.mef")
    return compare(rt, dir, "softmax", [[4, 8]], m, 1)


def case_mlp(rt: Runtime, dir: String) raises -> String:
    var g = Graph("mlp_mojo", [f32([16, 17]), f32([17, 32]), f32([32]), f32([32, 6]), f32([6])])
    var p = g.inputs()
    g.output([relu(p[0] @ p[1] + p[2]) @ p[3] + p[4]])
    var m = g.compile(rt, dir + "/mlp_mojo.mef")
    return compare(rt, dir, "mlp", [[16, 17], [17, 32], [32], [32, 6], [6]], m, 1)


def case_generated_only(rt: Runtime, dir: String) raises -> String:
    """Three ops no hand-written code mentions, straight from the generated
    layer."""
    var g = Graph("generated_only_mojo", [f32([4, 8])])
    var x = g.inputs()[0].copy()
    var b = PythonBackend.current()
    var top = ops.top_k(b, x, 3, -1)
    var cumsum = ops.mo_cumsum(b, x.type, x, 1, 0, 0)
    var erf = ops.mo_erf(b, x.type, x)
    g.output([top[0].copy(), top[1].copy(), cumsum^, erf^])
    var m = g.compile(rt, dir + "/generated_only_mojo.mef")
    return compare(rt, dir, "generated_only", [[4, 8]], m, 4)


def main() raises:
    var dir = String(argv()[1])
    var rt = Runtime()
    var failed = 0
    var names: List[String] = [
        "elementwise", "matmul", "reshape_transpose", "reductions", "softmax", "mlp",
        "generated_only",
    ]
    for name in names:
        try:
            var detail: String
            if name == "elementwise":
                detail = case_elementwise(rt, dir)
            elif name == "matmul":
                detail = case_matmul(rt, dir)
            elif name == "reshape_transpose":
                detail = case_reshape_transpose(rt, dir)
            elif name == "reductions":
                detail = case_reductions(rt, dir)
            elif name == "softmax":
                detail = case_softmax(rt, dir)
            elif name == "mlp":
                detail = case_mlp(rt, dir)
            else:
                detail = case_generated_only(rt, dir)
            print("PASS", name + ":", detail)
        except e:
            print("FAIL", name + ":", e)
            failed += 1
    if failed > 0:
        raise Error(String(failed) + " parity case(s) failed")
