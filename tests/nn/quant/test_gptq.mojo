"""Gates for `noeira/nn/quant/gptq.mojo`.

    pixi run python tools/quant/gptq_q4_0_reference.py --out tests/fixtures/quant
    pixi run mojo run -I . tests/nn/quant/test_gptq.mojo

[1] Parity with the numpy reference (the fixture) on its two cases: the
    objective within 1 %, and the codes and scales identical up to
    floating-point-order flips: at least 90 % of rows identical code for
    code, and every scale identical in those rows. The numpy reference
    itself agrees with the MLX pipeline it transcribes on every code and
    scale of both cases (checked once, outside this suite).
[2] A diagonal Hessian propagates no error (U is diagonal), so GPTQ must
    then BE RTN: codes and scales bit-identical to `quantize_rtn_q4_0`.
[3] `q4_0_objective` reproduces the reference's RTN objective, and GPTQ's
    objective is below RTN's on both cases.
[4] Rows are independent: GPTQ of the first 40 rows alone equals those rows
    of the full run (the threading never mixes rows), and two runs agree.
[5] k not a multiple of 32 raises.
"""

from std.math import abs
from std.random import random_float64, seed
from std.testing import assert_true

from noeira.io.safetensors import SafeTensors
from noeira.nn.quant.q4_0 import Q4_GROUP, quantize_rtn_q4_0
from noeira.nn.quant.gptq import gptq_q4_0, q4_0_objective

comptime FIX = "tests/fixtures/quant/gptq_q4_0.safetensors"


def _f64(ref v: List[Float32]) -> List[Float64]:
    var o = List[Float64](capacity=len(v))
    for x in v:
        o.append(Float64(x))
    return o^


def _bits(x: Float32) -> UInt32:
    return Scalar[DType.float32](x).to_bits[DType.uint32]()


def parity(mut f: SafeTensors, name: String) raises -> Bool:
    var shp = f.shape(name + ".w")
    var n = shp[0]
    var k = shp[1]
    var w = f.read_f32(name + ".w")
    var h = _f64(f.read_f32(name + ".h"))
    var q_ref = f.read_raw(name + ".q")
    var d_ref = f.read_f32(name + ".d")
    var obj_ref = f.read_f32(name + ".obj")
    var q = List[UInt8]()
    var d = List[Float32]()
    var st = gptq_q4_0(w, n, k, h.copy(), q, d)
    var groups = k // Q4_GROUP
    var codes_diff = 0
    var rows_same = 0
    var scale_bad = 0
    var first_min = k
    for r in range(n):
        var same = True
        for c in range(k):
            if q[r * k + c] != q_ref[r * k + c]:
                codes_diff += 1
                if same:
                    first_min = min(first_min, c)
                same = False
        if same:
            rows_same += 1
            for g in range(groups):
                if _bits(d[r * groups + g]) != _bits(d_ref[r * groups + g]):
                    scale_bad += 1
    var obj = q4_0_objective(w, q, d, n, k, h)
    var rel = abs(obj - Float64(obj_ref[0])) / Float64(obj_ref[0])
    print("    case", name, "[", n, "x", k, "]:", st.dead, "dead, damp", st.damp,
          "| codes differing", codes_diff, "of", n * k, "| rows identical",
          rows_same, "of", n, "| scales differing in those rows", scale_bad,
          "| objective", obj, "vs", obj_ref[0], "(rel", rel, ")",
          "| first differing column", first_min if codes_diff > 0 else -1)
    return rel < 0.01 and rows_same * 10 >= n * 9 and scale_bad == 0


def main() raises:
    var f = SafeTensors(String(FIX))
    print("GPTQ on the Q4_0 grid (CPU)")
    print("[1] parity with the numpy reference")
    for name in ["a", "b"]:
        assert_true(parity(f, String(name)), "GPTQ differs from the reference")

    print("[2] diagonal H: GPTQ = RTN bit for bit")
    seed(5)
    var n = 70
    var k = 288
    var w = List[Float32](capacity=n * k)
    for _ in range(n * k):
        w.append(Float32((random_float64() - 0.5) * 0.3))
    var hd = List[Float64](length=k * k, fill=0)
    for i in range(k):
        hd[i * k + i] = 0.1 + 10.0 * random_float64()
    var qg = List[UInt8]()
    var dg = List[Float32]()
    _ = gptq_q4_0(w, n, k, hd^, qg, dg)
    var qr = List[UInt8]()
    var dr = List[Float32]()
    quantize_rtn_q4_0(w, n, k, qr, dr)
    var nq = 0
    var nd = 0
    for i in range(n * k):
        nq += Int(qg[i] != qr[i])
    for i in range(len(dr)):
        nd += Int(_bits(dg[i]) != _bits(dr[i]))
    print("    [", n, "x", k, "] codes differing", nq, "| scales differing", nd)
    assert_true(nq == 0 and nd == 0, "GPTQ with a diagonal H is not RTN")

    print("[3] objectives")
    for name in ["a", "b"]:
        var nm = String(name)
        var shp = f.shape(nm + ".w")
        var wn = f.read_f32(nm + ".w")
        var h = _f64(f.read_f32(nm + ".h"))
        var obj_ref = f.read_f32(nm + ".obj")
        var q1 = List[UInt8]()
        var d1 = List[Float32]()
        quantize_rtn_q4_0(wn, shp[0], shp[1], q1, d1)
        var o_rtn = q4_0_objective(wn, q1, d1, shp[0], shp[1], h)
        var q2 = List[UInt8]()
        var d2 = List[Float32]()
        _ = gptq_q4_0(wn, shp[0], shp[1], h.copy(), q2, d2)
        var o_g = q4_0_objective(wn, q2, d2, shp[0], shp[1], h)
        var rel = abs(o_rtn - Float64(obj_ref[1])) / Float64(obj_ref[1])
        print("    case", nm, ": RTN", o_rtn, "(reference", obj_ref[1], ", rel", rel,
              ") | GPTQ", o_g, "| GPTQ / RTN", o_g / o_rtn)
        assert_true(rel < 1e-6, "q4_0_objective differs from the reference's")
        assert_true(o_g < 0.85 * o_rtn, "GPTQ does not beat RTN on its objective")

    print("[4] rows are independent; runs repeat")
    var wb = f.read_f32(String("b.w"))
    var hb = _f64(f.read_f32(String("b.h")))
    var kb = f.shape(String("b.w"))[1]
    var nb = f.shape(String("b.w"))[0]
    var qa = List[UInt8]()
    var da = List[Float32]()
    _ = gptq_q4_0(wb, nb, kb, hb.copy(), qa, da)
    var qb = List[UInt8]()
    var db = List[Float32]()
    _ = gptq_q4_0(wb, nb, kb, hb.copy(), qb, db)
    var sub = List[Float32](capacity=40 * kb)
    for i in range(40 * kb):
        sub.append(wb[i])
    var qs = List[UInt8]()
    var ds = List[Float32]()
    _ = gptq_q4_0(sub, 40, kb, hb.copy(), qs, ds)
    var rep = 0
    var subd = 0
    for i in range(nb * kb):
        rep += Int(qa[i] != qb[i])
    for i in range(40 * kb):
        subd += Int(qa[i] != qs[i])
    for i in range(40 * kb // Q4_GROUP):
        subd += Int(_bits(da[i]) != _bits(ds[i]))
    print("    two runs: codes differing", rep, "| first 40 rows alone vs the full run:", subd, "differences")
    assert_true(rep == 0 and subd == 0, "GPTQ depends on the run or on the other rows")

    print("[5] k not a multiple of 32 raises")
    var raised = False
    try:
        var q5 = List[UInt8]()
        var d5 = List[Float32]()
        _ = gptq_q4_0(w, 1, 40, List[Float64](length=1600, fill=1.0), q5, d5)
    except e:
        raised = True
        print("    raised:", e)
    assert_true(raised, "k = 40 was accepted")
    print("ALL PASS")
